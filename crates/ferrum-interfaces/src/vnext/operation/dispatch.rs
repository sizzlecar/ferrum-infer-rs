use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use super::super::{
    classify_device_error, BackingInitializationEncodeError, BatchInvocationId, BufferUsage,
    CompletionHandle, CompletionReaper, CompletionReservation,
    DefinitelyNotSubmittedRetryAuthority, DefinitelyNotSubmittedWaveRetryAuthority,
    DeviceBatchingForm, DeviceCommandBatch, DeviceCommandLogicalWork, DeviceComputePathRequirement,
    DeviceExecutionPath, DeviceReusableExecutionCapture, DeviceReusableExecutionInvocation,
    DeviceReusableExecutionProgram, DeviceReusableExecutionProgramId,
    DeviceReusableExecutionTopologyFingerprint, DeviceRuntime, DeviceSubmissionAttribution,
    DeviceSubmissionExecutionTiming, DeviceSubmissionStage, DeviceSubmissionTimingSink,
    DeviceTimingMeasurement, DeviceTimingMode, ExecutablePlanView, ExecutionIdentityEnvelope,
    ExecutionIdentityParts, ExecutionLane, HostTransferLayout, IdentifiedFailure,
    IndeterminateSubmissionHandle, InvocationResourceLease, LaneSubmitOutcome,
    LogicalBackingBufferView, NodeId, NodeInvocationId, ParticipantNodeKey,
    PreparedStepSubmissionWave, ProgramBindingNodeBinding, ResourceId, SpanId,
    StepParticipantFrameAssignment, SubmissionWavePurpose, TrustedActiveSequenceBinding,
    VNextError, EXECUTION_IDENTITY_VERSION,
};
use super::determinism::{
    SubmissionWaveDeterminismHandle, SubmissionWaveDeterminismReadbackPlan,
    SubmissionWaveDeterminismRestore,
};
use super::{
    encode_provider_workspace_initialization, encode_submission_wave_workspace_initializations,
    invalid_operation, BatchOperationIdentity, BatchOperationNodeIdentity,
    BatchOperationParticipantIdentity, BatchedOperationInvocation, BoundOperationProvider,
    ElementType, OperationFailure, OperationInvocation, OperationInvocationResources,
    ProviderReplayEquivalence, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, TensorAccess,
};

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
    fn new(
        batch_identity: BatchOperationIdentity,
        submission_fingerprint: String,
        device: DeviceSubmissionAttribution,
    ) -> Result<Self, VNextError> {
        if device.commands().iter().any(|command| {
            command.node_index().is_some_and(|node_index| {
                let Ok(node_index_usize) = usize::try_from(node_index) else {
                    return true;
                };
                if batch_identity.node_id_at(node_index_usize).is_none() {
                    return true;
                }
                u32::try_from(
                    batch_identity
                        .node_participant_count(node_index_usize)
                        .unwrap_or_default(),
                )
                .map_or(true, |count| count != command.participant_count())
            })
        }) {
            return Err(invalid_operation(
                "device submission attribution differs from batch node identity",
            ));
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
                || replayed_segment
                    .logical_commands()
                    .first()
                    .is_none_or(|command| {
                        program_id.immediate_sequences() != command.participant_count()
                            || program_id.immediate_tokens() != command.token_count()
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

struct DisabledSubmissionWaveDispatchTimingSink;

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

struct SubmissionWaveDispatchStageTimer<'sink, S>
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
    fn start(sink: &'sink S, stage: SubmissionWaveDispatchStage) -> Self {
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

/// The only public path from a resolved plan to an operation kernel.
fn validate_program_binding_patch(
    resolved: &dyn ExecutablePlanView,
    node_identity: &BatchOperationNodeIdentity,
    program_binding: Option<&ProgramBindingNodeBinding>,
    encoded_program_binding_count: usize,
    program_binding_resources: &mut BTreeSet<ResourceId>,
) -> Result<(), VNextError> {
    if program_binding.is_some() != (encoded_program_binding_count == 1) {
        return Err(invalid_operation(
            "compiled program binding slot and provider patch cardinality differ",
        ));
    }
    let Some(program_binding) = program_binding else {
        return Ok(());
    };
    let (plan_node_index, binding_resource) = resolved
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| node.id() == node_identity.node_id())
        .and_then(|(node_index, node)| {
            node.binding_resource()
                .map(|resource| (node_index, resource))
        })
        .ok_or_else(|| {
            invalid_operation("program binding requires a provider-owned binding workspace")
        })?;
    if program_binding.node_index() != plan_node_index
        || program_binding.slot().node_id() != node_identity.node_id()
        || program_binding.slot().resource_id() != binding_resource
    {
        return Err(invalid_operation(
            "provider program binding differs from its compiled node slot",
        ));
    }
    if !program_binding_resources.insert(binding_resource.clone()) {
        return Err(invalid_operation(
            "program bindings from different nodes alias one binding workspace",
        ));
    }
    Ok(())
}

pub struct OperationDispatch;

impl OperationDispatch {
    fn trusted_submission_node_identity(
        resolved: &dyn ExecutablePlanView,
        active: &TrustedActiveSequenceBinding,
        frame: StepParticipantFrameAssignment,
        node_index: usize,
    ) -> Result<ExecutionIdentityEnvelope, VNextError> {
        active.ensure_open_for_emission()?;
        let plan = resolved.execution_plan();
        let node_count = u64::try_from(plan.payload().nodes().len())
            .map_err(|_| invalid_operation("immutable plan node count exceeds u64"))?;
        let node_index = u64::try_from(node_index)
            .map_err(|_| invalid_operation("submission wave node index exceeds u64"))?;
        let node = plan
            .payload()
            .nodes()
            .get(node_index as usize)
            .ok_or_else(|| invalid_operation("submission wave node index is out of bounds"))?;
        let completed_frames = frame.frame_id().get() - 1;
        let node_invocation = completed_frames
            .checked_mul(node_count)
            .and_then(|value| value.checked_add(node_index))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_operation("node invocation id space is exhausted"))?;
        let node_invocation_id = NodeInvocationId::try_from(node_invocation)?;

        // RequestAccepted and PlanBuilt consume the first two journal rows.
        // Every immutable-plan frame then emits FrameStarted, three rows per
        // node, and FrameCompleted. This operation identity is therefore the
        // exact future OperationSubmitted row and remains stable across a
        // definitely-not-submitted retry of the same frame.
        let events_per_frame = node_count
            .checked_mul(3)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| invalid_operation("execution event sequence space is exhausted"))?;
        let sequence = completed_frames
            .checked_mul(events_per_frame)
            .and_then(|value| value.checked_add(node_index.checked_mul(3)?))
            .and_then(|value| value.checked_add(5))
            .ok_or_else(|| invalid_operation("execution event sequence space is exhausted"))?;

        let span_root = format!("vnext/request/{}", active.fingerprint());
        let node_span = SpanId::new(format!(
            "{span_root}/frame/{}/node/{node_invocation}",
            frame.frame_id()
        ))?;
        let operation_span = SpanId::new(format!("{node_span}/operation"))?;
        let provisioning = active.static_provisioning_identity();
        ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: active.run_id().clone(),
            request_id: active.request_id().clone(),
            sequence,
            plan_id: Some(plan.payload().plan_id().clone()),
            plan_hash: Some(plan.plan_hash().clone()),
            frame_id: Some(frame.frame_id()),
            node_invocation_id: Some(node_invocation_id),
            node_id: Some(node.id().clone()),
            operation_id: Some(node.operation_id().clone()),
            provider_id: Some(node.selection().selected_provider().clone()),
            device_id: Some(plan.payload().device_id().clone()),
            resource_pool_id: active.static_pool_id(),
            resource_pool_identity_fingerprint: active
                .static_pool_identity_fingerprint_ref()
                .map(str::to_owned),
            provisioning_run_id: provisioning.map(|identity| identity.run_id().clone()),
            provisioning_request_id: provisioning.map(|identity| identity.request_id().clone()),
            transaction_id: provisioning.map(|identity| identity.transaction_id().clone()),
            active_sequence_slot: Some(active.sequence_authority().sparse_id()),
            admission_generation: Some(active.sequence_authority().generation()),
            activation_epoch: Some(active.activation_epoch()),
            runtime_implementation_fingerprint: Some(
                active.runtime_implementation_fingerprint().to_owned(),
            ),
            active_sequence_fingerprint: Some(active.fingerprint().to_owned()),
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id: operation_span,
            parent_span_id: Some(node_span),
            async_links: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_node_identity<'binding, R, I>(
        resolved: &dyn ExecutablePlanView,
        participant_identities: Vec<ExecutionIdentityEnvelope>,
        active_bindings: I,
        resources: OperationInvocationResources<'_, R>,
        lane: &Arc<ExecutionLane<R>>,
        node_index: u32,
        participant_start: u32,
    ) -> Result<BatchOperationNodeIdentity, VNextError>
    where
        R: DeviceRuntime,
        I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let plan = resolved.execution_plan();
        let node_id = resources.node_id()?;
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        // Wave construction proves every node shares this authority by Arc identity.
        let validate_common_authority = match resources {
            OperationInvocationResources::Invocation(_) => true,
            OperationInvocationResources::Wave {
                node_index: resource_node_index,
                ..
            } => resource_node_index == 0,
        };
        let participant_count = resources.participant_count()?;
        let frames = resources.participant_frames()?;
        let plan_identity_matches = validate_common_authority
            .then(|| {
                resources.plan_identity_matches(
                    plan.payload().plan_id(),
                    plan.plan_hash(),
                    plan.payload().device_id(),
                )
            })
            .transpose()?;
        if participant_identities.is_empty()
            || participant_identities.len() != participant_count
            || participant_identities.len() != frames.len()
            || participant_identities.len() != active_bindings.len()
            || resources.prepared_participant_count()? != participant_count
            || validate_common_authority && {
                !plan_identity_matches.expect("common wave authority requested plan evidence")
                    || !Arc::ptr_eq(resources.runtime(), lane.runtime_arc())
                    || resources.step_resources().execution_lane().id() != lane.id()
                    || lane.descriptor() != resolved.device()
                    || lane.descriptor() != resolved.capabilities().device()
                    || lane.descriptor().runtime_implementation_fingerprint
                        != plan.payload().device_runtime_implementation_fingerprint()
            }
        {
            return Err(invalid_operation(
                "batch node identity inputs differ from submission resources, plan, or lane",
            ));
        }
        let mut participant_projections = Vec::with_capacity(participant_identities.len());
        for (local_index, (identity, active)) in participant_identities
            .into_iter()
            .zip(active_bindings)
            .enumerate()
        {
            let participant = resources.participant(local_index)?;
            let frame = *frames
                .get(local_index)
                .ok_or_else(|| invalid_operation("operation participant frame is missing"))?;
            let session = validate_common_authority
                .then(|| resources.participant_session_identity(local_index))
                .transpose()?;
            let key =
                ParticipantNodeKey::new(frame.participant(), frame.frame_id(), node.id().clone());
            let parts = identity.parts();
            if key.sequence_authority() != participant.sequence_authority()
                || key.request_authority() != participant.request_authority()
                || key.frame_id() != frame.frame_id()
                || validate_common_authority && {
                    let session = session
                        .as_ref()
                        .expect("common wave authority requested session evidence");
                    active.sequence_authority() != participant.sequence_authority()
                        || active.coordinator_id() != resources.coordinator_id()?
                        || active.run_id() != participant.run_id()
                        || active.request_id() != participant.request_id()
                        || !active.matches_sequence_session(session.0, session.1)
                        || active.plan().plan_id() != plan.payload().plan_id()
                        || active.plan().plan_hash() != plan.plan_hash()
                        || active.plan().device_id() != plan.payload().device_id()
                        || active.runtime_implementation_fingerprint()
                            != plan.payload().device_runtime_implementation_fingerprint()
                }
                || parts.run_id != *active.run_id()
                || parts.request_id != *active.request_id()
                || parts.plan_id.as_ref() != Some(plan.payload().plan_id())
                || parts.plan_hash.as_ref() != Some(plan.plan_hash())
                || parts.frame_id != Some(frame.frame_id())
                || parts.node_invocation_id.is_none()
                || parts.node_id.as_ref() != Some(node.id())
                || parts.operation_id.as_ref() != Some(node.operation_id())
                || parts.provider_id.as_ref() != Some(node.selection().selected_provider())
                || parts.device_id.as_ref() != Some(plan.payload().device_id())
                || parts.active_sequence_slot != Some(active.sequence_authority().sparse_id())
                || parts.admission_generation != Some(active.sequence_authority().generation())
                || parts.activation_epoch != Some(active.activation_epoch())
                || parts.runtime_implementation_fingerprint.as_deref()
                    != Some(active.runtime_implementation_fingerprint())
                || parts.active_sequence_fingerprint.as_deref() != Some(active.fingerprint())
                || parts.completed_sequence_fingerprint.is_some()
                || parts.aborted_sequence_fingerprint.is_some()
                || parts.resource_id.is_some()
                || parts.resource_generation.is_some()
                || parts.resource_batch_fingerprint.is_some()
            {
                return Err(invalid_operation(format!(
                    "batch node {node_index} participant {local_index} differs from its resource, frame, session, or plan identity"
                )));
            }
            let local_index = u32::try_from(local_index)
                .map_err(|_| invalid_operation("batch participant index exceeds u32"))?;
            participant_projections.push(BatchOperationParticipantIdentity::new(
                participant_start.checked_add(local_index).ok_or_else(|| {
                    invalid_operation("physical batch participant index overflows u32")
                })?,
                key,
                identity,
            ));
        }
        BatchOperationNodeIdentity::from_validated(
            node_index,
            node.id().clone(),
            node.operation_id().clone(),
            node.selection().selected_provider().clone(),
            node.provider_execution_semantics(),
            resources.work_shape()?.fingerprint().to_owned(),
            participant_projections,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_batch_identity<R>(
        resolved: &dyn ExecutablePlanView,
        participant_identities: Vec<ExecutionIdentityEnvelope>,
        active_bindings: &[TrustedActiveSequenceBinding],
        invocation_resources: &InvocationResourceLease<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        let resources = OperationInvocationResources::Invocation(invocation_resources);
        let node_identity = Self::bind_node_identity(
            resolved,
            participant_identities,
            active_bindings.iter(),
            resources,
            lane,
            0,
            0,
        )?;
        BatchOperationIdentity::from_validated(
            resources.batch_step_id(),
            resources.batch_invocation_id(),
            plan.payload().plan_id().clone(),
            plan.plan_hash().clone(),
            plan.payload().device_id().clone(),
            plan.payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            lane.id(),
            resources.backing_fingerprint().to_owned(),
            vec![node_identity],
        )
    }

    fn bind_submission_wave_identity_from_envelopes<'binding, R, I>(
        resolved: &dyn ExecutablePlanView,
        participant_identities: Vec<Vec<ExecutionIdentityEnvelope>>,
        active_bindings: I,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let plan = resolved.execution_plan();
        if wave.nodes().is_empty()
            || wave.execution_lane_id() != lane.id()
            || participant_identities.len() != wave.nodes().len()
            || active_bindings.len() == 0
            || wave.claimed_backing().node_count() != wave.nodes().len()
            || wave.claimed_backing().plan_node_count() != plan.payload().nodes().len()
            || wave
                .nodes()
                .windows(2)
                .any(|pair| pair[0].plan_node_index() >= pair[1].plan_node_index())
            || wave.nodes().iter().any(|prepared| {
                plan.payload()
                    .nodes()
                    .get(prepared.plan_node_index())
                    .is_none_or(|planned| prepared.node_id() != planned.id())
            })
        {
            return Err(invalid_operation(
                "submission wave identity must cover one exact canonical immutable-plan node scope",
            ));
        }
        let mut participant_start = 0_u32;
        let mut nodes = Vec::with_capacity(wave.nodes().len());
        for (node_index, (identities, _node)) in participant_identities
            .into_iter()
            .zip(wave.nodes())
            .enumerate()
        {
            let node_index = u32::try_from(node_index)
                .map_err(|_| invalid_operation("submission wave node index exceeds u32"))?;
            let participant_count = u32::try_from(identities.len())
                .map_err(|_| invalid_operation("submission wave participant count exceeds u32"))?;
            let node_identity = Self::bind_node_identity(
                resolved,
                identities,
                active_bindings.clone(),
                OperationInvocationResources::Wave {
                    wave,
                    node_index: node_index as usize,
                },
                lane,
                node_index,
                participant_start,
            )?;
            participant_start = participant_start
                .checked_add(participant_count)
                .ok_or_else(|| {
                    invalid_operation("submission wave participant index space overflows u32")
                })?;
            nodes.push(node_identity);
        }
        BatchOperationIdentity::from_validated(
            wave.batch_step_id(),
            wave.batch_invocation_id(),
            plan.payload().plan_id().clone(),
            plan.plan_hash().clone(),
            plan.payload().device_id().clone(),
            plan.payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            lane.id(),
            wave.fingerprint().to_owned(),
            nodes,
        )
    }

    /// Binds one immutable-plan submission wave without accepting caller-made
    /// execution envelopes. Frame, node, provider, provisioning, span, and
    /// invocation identities are minted from core-owned plan/session evidence.
    pub fn bind_submission_wave_identity<'binding, R, I>(
        resolved: &dyn ExecutablePlanView,
        active_bindings: I,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        if active_bindings.len() == 0 {
            return Err(invalid_operation(
                "submission wave requires a non-empty active participant set",
            ));
        }
        let participant_identities = wave
            .nodes()
            .iter()
            .enumerate()
            .map(|(node_index, node)| {
                if active_bindings.len() != node.participant_frames().len() {
                    return Err(invalid_operation(format!(
                        "submission wave node {node_index} active binding count differs from its participant frames"
                    )));
                }
                node.participant_frames()
                    .iter()
                    .copied()
                    .zip(active_bindings.clone())
                    .map(|(frame, active)| {
                        Self::trusted_submission_node_identity(
                            resolved,
                            active,
                            frame,
                            node.plan_node_index(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::bind_submission_wave_identity_from_envelopes(
            resolved,
            participant_identities,
            active_bindings,
            wave,
            lane,
        )
    }

    /// Derives the exact reusable-program identity for the current wave without
    /// materializing device buffers or entering dispatch.
    ///
    /// Provider topology remains opaque. Core binds each dynamic row to its
    /// immutable node/provider position before aggregating it, so two providers
    /// cannot accidentally alias the same program variant. One ineligible
    /// provider vetoes resident reuse for the complete wave.
    pub fn reusable_execution_program_id_for_wave<R>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<Option<DeviceReusableExecutionProgramId>, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        let plan_nodes = plan.payload().nodes();
        if providers.is_empty()
            || providers.len() != wave.nodes().len()
            || wave.execution_lane_id() != lane.id()
            || lane.descriptor() != resolved.device()
        {
            return Err(invalid_operation(
                "reusable execution topology requires one exact provider per wave node and lane",
            ));
        }
        let Some(program_id) = wave.claimed_backing().reusable_execution_program_id(
            &lane.descriptor().runtime_implementation_fingerprint,
            lane.id(),
        )?
        else {
            return Ok(None);
        };

        const DOMAIN: &[u8] = b"ferrum.runtime-vnext.reusable-program-topology.v3\0";
        let mut digest = Sha256::new();
        digest.update(DOMAIN);
        digest.update(
            u64::try_from(wave.nodes().len())
                .map_err(|_| invalid_operation("reusable topology node count exceeds u64"))?
                .to_le_bytes(),
        );
        for (wave_node_index, (provider, prepared_node)) in
            providers.iter().zip(wave.nodes()).enumerate()
        {
            let plan_node_index = prepared_node.plan_node_index();
            let node = plan_nodes.get(plan_node_index).ok_or_else(|| {
                invalid_operation("reusable execution wave node is absent from the immutable plan")
            })?;
            if provider.plan_id != *plan.payload().plan_id()
                || provider.plan_hash != *plan.plan_hash()
                || provider.node_id != *node.id()
                || prepared_node.node_id() != node.id()
            {
                return Err(invalid_operation(
                    "reusable execution topology provider or node differs from the immutable plan",
                ));
            }
            let request = ReusableExecutionTopologyRequest::new(
                node.id(),
                node.operation_id(),
                node.attributes(),
                node.values(),
                plan.payload().memory(),
                prepared_node.work_shape(),
                wave.claimed_backing(),
                wave.step_resources().backing_slices(),
            )?;
            let declared = node.provider_execution_semantics().replay_equivalence();
            let dynamic_topology = match (
                declared,
                provider.provider().reusable_execution_topology(request)?,
            ) {
                (
                    ProviderReplayEquivalence::BitwiseEagerEquivalent,
                    ReusableExecutionTopology::Static,
                ) => None,
                (
                    ProviderReplayEquivalence::BitwiseEagerEquivalent,
                    ReusableExecutionTopology::Dynamic(topology),
                ) => Some(topology),
                (_, ReusableExecutionTopology::Ineligible) => return Ok(None),
                (
                    ProviderReplayEquivalence::Ineligible,
                    ReusableExecutionTopology::Static | ReusableExecutionTopology::Dynamic(_),
                ) => {
                    return Err(invalid_operation(format!(
                        "provider `{}` returned reusable topology without a bitwise eager-equivalence contract",
                        provider.descriptor().provider_id()
                    )))
                }
            };
            let (topology_kind, topology_bytes) = dynamic_topology
                .as_ref()
                .map_or((0_u8, &[][..]), |topology| (1_u8, topology.as_bytes()));
            let wave_node_index = u64::try_from(wave_node_index)
                .map_err(|_| invalid_operation("reusable topology wave node index exceeds u64"))?;
            let plan_node_index = u64::try_from(plan_node_index)
                .map_err(|_| invalid_operation("reusable topology plan node index exceeds u64"))?;
            let node_id = node.id().as_str().as_bytes();
            let provider_id = provider.descriptor().provider_id().as_str().as_bytes();
            let node_id_len = u64::try_from(node_id.len())
                .map_err(|_| invalid_operation("reusable topology node id exceeds u64"))?;
            let provider_id_len = u64::try_from(provider_id.len())
                .map_err(|_| invalid_operation("reusable topology provider id exceeds u64"))?;
            let topology_len = u64::try_from(topology_bytes.len())
                .map_err(|_| invalid_operation("reusable topology payload exceeds u64"))?;
            digest.update(wave_node_index.to_le_bytes());
            digest.update(plan_node_index.to_le_bytes());
            digest.update(node_id_len.to_le_bytes());
            digest.update(node_id);
            digest.update(provider_id_len.to_le_bytes());
            digest.update(provider_id);
            digest.update([topology_kind]);
            digest.update(topology_len.to_le_bytes());
            digest.update(topology_bytes);
        }
        Ok(Some(program_id.with_topology_fingerprint(
            DeviceReusableExecutionTopologyFingerprint::from_sha256(digest.finalize().into()),
        )))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit<R>(
        provider: &BoundOperationProvider<'_, R>,
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: &[TrustedActiveSequenceBinding],
        mut invocation_resources: InvocationResourceLease<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, OperationDispatchError<R>>
    where
        R: DeviceRuntime,
    {
        let node_identity = batch_identity.single_node().ok_or_else(|| {
            OperationDispatchError::Contract(invalid_operation(
                "single-operation dispatch requires a one-node batch identity",
            ))
        })?;
        provider
            .validate_binding(resolved, node_identity.node_id())
            .map_err(OperationDispatchError::Contract)?;
        if active_bindings.is_empty()
            || active_bindings.len() != batch_identity.participants().len()
            || lane.id() != batch_identity.lane_id()
            || lane.descriptor().id != *batch_identity.device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != batch_identity.runtime_implementation_fingerprint()
        {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation execution lane or participant set differs from batch identity",
            )));
        }
        invocation_resources
            .begin_dispatch()
            .map_err(OperationDispatchError::Contract)?;
        let mut completion = CompletionReaper::reserve(
            reaper,
            invocation_resources,
            Arc::clone(lane),
            batch_identity.clone(),
        )
        .map_err(OperationDispatchError::Contract)?;
        let runtime = lane.runtime();
        if !lane.current_descriptor_matches_snapshot() {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation encode runtime differs from its execution lane snapshot",
            )));
        }
        let mut commands = DeviceCommandBatch::with_capacity(3);
        completion
            .encode_backing_initializations(runtime, &mut commands)
            .map_err(|error| map_backing_initialization_error(runtime, batch_identity, error))?;
        let invocation = BatchedOperationInvocation::from_resolved(
            runtime,
            resolved,
            provider.dispatch(),
            batch_identity,
            completion.invocation(),
            active_bindings,
        )
        .map_err(OperationDispatchError::Contract)?;
        let plan_node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_identity.node_id())
            .ok_or_else(|| {
                OperationDispatchError::Contract(invalid_operation(
                    "operation workspace node is absent from the immutable plan",
                ))
            })?;
        if let Some(requirement) = plan_node.provider_resources().scratch() {
            let scratch_view = invocation
                .participants()
                .first()
                .and_then(OperationInvocation::scratch_view)
                .ok_or_else(|| {
                    OperationDispatchError::Contract(invalid_operation(
                        "operation scratch requirement has no invocation view",
                    ))
                })?;
            encode_provider_workspace_initialization::<R, DefinitelyNotSubmittedRetryAuthority<R>>(
                runtime,
                0,
                node_identity,
                requirement,
                invocation.work_shape().resource_work(),
                scratch_view,
                SubmissionScratchInitialization::ProviderContract,
                &mut commands,
            )?;
        }
        let expected_phase = invocation.operation().profile_phase;
        let operation = match provider.provider().encode_selected(invocation) {
            Ok(operation) => operation,
            Err(failure)
                if batch_identity.contains_identity(failure.identity())
                    && failure.phase() == expected_phase =>
            {
                return Err(OperationDispatchError::Provider(failure));
            }
            Err(_) => {
                return Err(OperationDispatchError::Contract(invalid_operation(
                    "operation provider returned a failure for a different execution identity or profile phase",
                )));
            }
        };
        if !lane.current_descriptor_matches_snapshot() {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation encode completion runtime drifted",
            )));
        }
        commands.push_operation(0, operation);
        let timing_mode = commands.timing_mode();
        let mut lane_reservation = lane
            .reserve_enqueue()
            .map_err(OperationDispatchError::Contract)?;
        completion.mark_submission_started();
        match lane_reservation.submit(commands) {
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                drop(lane_reservation);
                let retry = completion
                    .definitely_not_submitted()
                    .map_err(OperationDispatchError::Contract)?;
                let failures = batch_identity
                    .participants()
                    .iter()
                    .map(|participant| {
                        classify_device_error(runtime, participant.identity().clone(), &error)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(OperationDispatchError::Contract)?;
                Err(OperationDispatchError::DefinitelyNotSubmitted { failures, retry })
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                drop(lane_reservation);
                let recovery = completion.submission_indeterminate();
                Err(OperationDispatchError::SubmissionIndeterminate { recovery })
            }
            LaneSubmitOutcome::Submitted(fence) => {
                drop(lane_reservation);
                let completion = match completion.arm(fence, timing_mode) {
                    Ok(completion) => completion,
                    Err((error, completion)) => {
                        return Err(OperationDispatchError::PostSubmitContract {
                            error,
                            completion,
                        });
                    }
                };
                if !lane.current_descriptor_matches_snapshot() {
                    lane.fail_closed();
                    return Err(OperationDispatchError::PostSubmitContract {
                        error: invalid_operation("operation submit completion runtime drifted"),
                        completion,
                    });
                }
                Ok(completion)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_wave<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::encode_and_submit_wave_with_inputs(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            &[],
            wave,
            lane,
            reaper,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_wave_with_inputs<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            SubmissionExecutionPolicy::adaptive(),
            None,
            None,
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )
        .map(|profiled| profiled.into_parts().0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_wave_with_inputs_and_policy<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        execution_policy: SubmissionExecutionPolicy,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            execution_policy,
            None,
            None,
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )
        .map(|profiled| profiled.into_parts().0)
    }

    /// Dispatches one prepared wave while attributing host time to exact typed
    /// ownership boundaries. The diagnostic sink receives no access to the
    /// command, submission, completion, or failure value.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_wave_with_inputs_and_timing<'binding, R, I, S>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        execution_policy: SubmissionExecutionPolicy,
        timing_sink: &S,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<ProfiledSubmissionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
        S: SubmissionWaveDispatchTimingSink,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            execution_policy,
            None,
            None,
            timing_sink,
            wave,
            lane,
            reaper,
        )
    }

    /// Executes one complete plan-derived determinism restore through eager
    /// provider commands. The returned profiled handle retains physical path
    /// attribution for the hardware artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_determinism_eager_wave<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        restore: &SubmissionWaveDeterminismRestore,
        scratch_fill: u8,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<SubmissionWaveDeterminismHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let readback_plan = SubmissionWaveDeterminismReadbackPlan::from_restore(
            resolved,
            batch_identity,
            &wave,
            restore,
        )
        .map_err(SubmissionWaveDispatchError::Contract)?;
        let restore_fingerprint = restore
            .logical_fingerprint()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let profiled = Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            &[],
            SubmissionExecutionPolicy::determinism_eager(scratch_fill),
            Some(restore),
            None,
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )?;
        Ok(SubmissionWaveDeterminismHandle::from_profiled(
            profiled,
            readback_plan,
            restore_fingerprint,
            DeviceExecutionPath::Eager,
        ))
    }

    /// Executes the same complete restore through one sealed resident replay
    /// program. No adaptive capture or eager fallback is permitted.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_determinism_replayed_wave<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        restore: &SubmissionWaveDeterminismRestore,
        scratch_fill: u8,
        reusable_program: &DeviceReusableExecutionProgram,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<SubmissionWaveDeterminismHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let readback_plan = SubmissionWaveDeterminismReadbackPlan::from_restore(
            resolved,
            batch_identity,
            &wave,
            restore,
        )
        .map_err(SubmissionWaveDispatchError::Contract)?;
        let restore_fingerprint = restore
            .logical_fingerprint()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let profiled = Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            &[],
            SubmissionExecutionPolicy::determinism_replayed(scratch_fill),
            Some(restore),
            Some(reusable_program),
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )?;
        Ok(SubmissionWaveDeterminismHandle::from_profiled(
            profiled,
            readback_plan,
            restore_fingerprint,
            DeviceExecutionPath::Replayed,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_reusable_wave_with_inputs<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        reusable_program: &DeviceReusableExecutionProgram,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            SubmissionExecutionPolicy::adaptive(),
            None,
            Some(reusable_program),
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )
        .map(|profiled| profiled.into_parts().0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_reusable_wave_with_inputs_and_policy<'binding, R, I>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        reusable_program: &DeviceReusableExecutionProgram,
        execution_policy: SubmissionExecutionPolicy,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            execution_policy,
            None,
            Some(reusable_program),
            &DisabledSubmissionWaveDispatchTimingSink,
            wave,
            lane,
            reaper,
        )
        .map(|profiled| profiled.into_parts().0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_reusable_wave_with_inputs_and_timing<'binding, R, I, S>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        reusable_program: &DeviceReusableExecutionProgram,
        execution_policy: SubmissionExecutionPolicy,
        timing_sink: &S,
        wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<ProfiledSubmissionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
        S: SubmissionWaveDispatchTimingSink,
    {
        Self::encode_and_submit_wave_with_inputs_timed(
            providers,
            resolved,
            batch_identity,
            active_bindings,
            timing_mode,
            input_uploads,
            execution_policy,
            None,
            Some(reusable_program),
            timing_sink,
            wave,
            lane,
            reaper,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_and_submit_wave_with_inputs_timed<'binding, R, I, S>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        timing_mode: DeviceTimingMode,
        input_uploads: &[SubmissionWaveInputUpload],
        execution_policy: SubmissionExecutionPolicy,
        determinism_restore: Option<&SubmissionWaveDeterminismRestore>,
        reusable_program: Option<&DeviceReusableExecutionProgram>,
        timing_sink: &S,
        mut wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<ProfiledSubmissionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
        S: SubmissionWaveDispatchTimingSink,
    {
        let contract_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::ContractValidateAndReserve,
        );
        let active_participant_count = active_bindings.len();
        if providers.is_empty()
            || providers.len() != wave.nodes().len()
            || providers.len() != batch_identity.node_count()
            || active_participant_count == 0
            || batch_identity.batch_step_id() != wave.batch_step_id()
            || batch_identity.batch_invocation_id() != wave.batch_invocation_id()
            || batch_identity.claimed_backing_fingerprint() != wave.fingerprint()
            || lane.id() != batch_identity.lane_id()
            || lane.descriptor().id != *batch_identity.device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != batch_identity.runtime_implementation_fingerprint()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave execution lane, resources, nodes, or participants differ from batch identity",
            )));
        }
        if !matches!(
            (wave.purpose(), determinism_restore.is_some()),
            (SubmissionWavePurpose::FullPlan, false)
                | (SubmissionWavePurpose::DeterminismProbe, true)
        ) {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission wave purpose differs from its product or determinism dispatch path",
            )));
        }
        if let Some(restore) = determinism_restore {
            restore
                .validate_for_submission(
                    lane.runtime(),
                    providers,
                    resolved,
                    batch_identity,
                    active_bindings.clone(),
                    &wave,
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
            if !input_uploads.is_empty()
                || usize::try_from(restore.participant_count()).ok()
                    != Some(active_participant_count)
                || execution_policy.compute_path() == DeviceComputePathRequirement::Adaptive
                || execution_policy.scratch_initialization()
                    == SubmissionScratchInitialization::ProviderContract
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "determinism submission requires complete restore coverage, explicit scratch fill, and one forced compute path",
                )));
            }
        }
        for (node_index, (provider, prepared_node)) in
            providers.iter().zip(wave.nodes()).enumerate()
        {
            let node_id = batch_identity.node_id_at(node_index).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "physical batch is missing a compiled plan node",
                ))
            })?;
            provider
                .validate_binding(resolved, node_id)
                .map_err(SubmissionWaveDispatchError::Contract)?;
            if prepared_node.node_id() != node_id
                || prepared_node.work_shape().fingerprint()
                    != batch_identity
                        .work_shape_fingerprint_at(node_index)
                        .expect("compiled physical batch node has work identity")
                || Some(provider.descriptor().provider_id())
                    != batch_identity.provider_id_at(node_index)
                || Some(provider.descriptor().operation_id())
                    != batch_identity.operation_id_at(node_index)
                || batch_identity.node_participant_count(node_index)
                    != Some(
                        usize::try_from(prepared_node.participant_count())
                            .expect("prepared wave participant count fits usize"),
                    )
                || batch_identity.node_participant_count(node_index)
                    != Some(active_participant_count)
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "wave provider or node identity differs from its prepared node",
                )));
            }
        }
        let reusable_execution_program_id =
            Self::reusable_execution_program_id_for_wave(providers, resolved, &wave, lane)
                .map_err(SubmissionWaveDispatchError::Contract)?;
        if let Some(reusable_program) = reusable_program {
            if !timing_mode.direct_reusable_execution_allowed()
                || execution_policy.compute_path() == DeviceComputePathRequirement::EagerOnly
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission timing or compute-path policy requires eager provider encoding",
                )));
            }
            let actual_program_id = reusable_execution_program_id.as_ref().ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "reusable execution program has no live program binding authority",
                ))
            })?;
            if actual_program_id != reusable_program.program_id()
                || reusable_program.segments().iter().any(|segment| {
                    segment.end_node_index() as usize > providers.len()
                        || segment.logical_command_count()
                            != segment.end_node_index() - segment.start_node_index()
                })
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "reusable execution program differs from the exact wave topology",
                )));
            }
        } else if execution_policy.compute_path() == DeviceComputePathRequirement::ReplayedOnly {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "replayed-only submission requires one sealed reusable execution program",
            )));
        }
        wave.begin_dispatch()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let mut completion =
            CompletionReaper::reserve_wave(reaper, wave, Arc::clone(lane), batch_identity.clone())
                .map_err(SubmissionWaveDispatchError::Contract)?;
        let runtime = lane.runtime();
        if !lane.current_descriptor_matches_snapshot() {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave encode runtime differs from its execution lane snapshot",
            )));
        }
        drop(contract_stage);

        let backing_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::BackingAndInputEncode,
        );
        let restore_capacity = determinism_restore.map_or(0, |restore| {
            restore
                .initializations()
                .len()
                .saturating_mul(active_participant_count)
        });
        let mut commands = DeviceCommandBatch::with_capacity_timing_and_compute_path(
            providers
                .len()
                .saturating_add(input_uploads.len())
                .saturating_add(restore_capacity),
            timing_mode,
            execution_policy.compute_path(),
        );
        if determinism_restore.is_some() {
            commands.require_logical_execution_path_attribution();
        }
        let backing_initialization_command_count = completion
            .encode_backing_initializations(runtime, &mut commands)
            .map_err(|error| map_backing_initialization_error(runtime, batch_identity, error))?;
        let workspace_initialization_command_count =
            encode_submission_wave_workspace_initializations(
                runtime,
                resolved,
                batch_identity,
                execution_policy.scratch_initialization(),
                &completion,
                &mut commands,
            )?;
        if commands.len()
            != backing_initialization_command_count
                .saturating_add(workspace_initialization_command_count)
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "backing initialization command accounting differs from encoded commands",
            )));
        }
        if let Some(restore) = determinism_restore {
            let restore_start = commands.len();
            let restore_command_count = encode_submission_wave_determinism_restore(
                runtime,
                resolved,
                batch_identity,
                &completion,
                restore,
                &mut commands,
            )?;
            if commands.len()
                != restore_start
                    .checked_add(restore_command_count)
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "determinism restore command accounting overflows usize",
                        ))
                    })?
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "determinism restore command accounting differs from encoded commands",
                )));
            }
        }
        encode_submission_wave_inputs(
            runtime,
            resolved,
            batch_identity,
            &completion,
            input_uploads,
            &mut commands,
        )?;
        drop(backing_stage);

        let provider_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::ProviderNodeEncode,
        );
        let pre_provider_command_count = commands.len();
        let mut encoded_provider_command_count = 0_usize;
        let mut program_bindings = Vec::new();
        let mut program_binding_resources = BTreeSet::new();
        let mut reusable_execution_binding_nodes = Vec::new();
        let mut encoded_operations = Vec::with_capacity(providers.len());
        if let Some(reusable_program) = reusable_program {
            let mut node_index = 0_usize;
            let mut segment_index = 0_usize;
            while node_index < providers.len() {
                let segment = reusable_program
                    .segments()
                    .get(segment_index)
                    .filter(|segment| segment.start_node_index() as usize == node_index);
                if let Some(segment) = segment {
                    let mut segment_dynamic_bindings = Vec::new();
                    let mut segment_result_bindings = Vec::new();
                    for binding_node_index in reusable_program
                        .per_wave_binding_node_indices()
                        .iter()
                        .copied()
                        .filter(|binding_node_index| segment.contains_node(*binding_node_index))
                    {
                        let binding_node_index =
                            usize::try_from(binding_node_index).map_err(|_| {
                                SubmissionWaveDispatchError::Contract(invalid_operation(
                                    "reusable execution binding node exceeds usize",
                                ))
                            })?;
                        let provider = &providers[binding_node_index];
                        let node_identity = batch_identity
                            .materialize_node(binding_node_index)
                            .map_err(SubmissionWaveDispatchError::Contract)?;
                        let invocation = BatchedOperationInvocation::from_wave_node(
                            runtime,
                            resolved,
                            provider.dispatch(),
                            batch_identity,
                            node_identity,
                            completion.wave(),
                            binding_node_index,
                            active_bindings.clone(),
                        )
                        .map_err(SubmissionWaveDispatchError::Contract)?;
                        let expected_phase = invocation.operation().profile_phase;
                        let program_binding = invocation.program_binding().cloned();
                        let bindings = match provider
                            .provider()
                            .encode_reusable_execution_bindings(invocation)
                        {
                            Ok(bindings) => bindings,
                            Err(failure)
                                if node_identity.contains_identity(failure.identity())
                                    && failure.phase() == expected_phase =>
                            {
                                return Err(SubmissionWaveDispatchError::Provider(failure));
                            }
                            Err(_) => {
                                return Err(SubmissionWaveDispatchError::Contract(
                                    invalid_operation(
                                        "reusable provider returned a failure for another node identity or profile phase",
                                    ),
                                ));
                            }
                        };
                        let program_binding_count = bindings.program_binding_count();
                        validate_program_binding_patch(
                            resolved,
                            node_identity,
                            program_binding.as_ref(),
                            program_binding_count,
                            &mut program_binding_resources,
                        )
                        .map_err(SubmissionWaveDispatchError::Contract)?;
                        let (mut node_program_bindings, mut dynamic_bindings, mut result_bindings) =
                            bindings.into_parts();
                        program_bindings.append(&mut node_program_bindings);
                        segment_dynamic_bindings.append(&mut dynamic_bindings);
                        segment_result_bindings.append(&mut result_bindings);
                    }
                    let invocation = DeviceReusableExecutionInvocation::new(
                        reusable_program.program_id().clone(),
                        segment.clone(),
                        u32::try_from(active_participant_count).map_err(|_| {
                            SubmissionWaveDispatchError::Contract(invalid_operation(
                                "reusable execution participant count exceeds u32",
                            ))
                        })?,
                        completion
                            .wave()
                            .claimed_backing()
                            .work_shape()
                            .immediate_tokens(),
                    )
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                    let compute = runtime
                        .encode_reusable_execution(invocation)
                        .map_err(|error| {
                            SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                                "device runtime rejected a sealed reusable program: {error}"
                            )))
                        })?
                        .ok_or_else(|| {
                            SubmissionWaveDispatchError::Contract(invalid_operation(
                                "device runtime published a reusable program it cannot encode",
                            ))
                        })?;
                    let operation_command_count = segment_dynamic_bindings
                        .len()
                        .checked_add(1)
                        .and_then(|count| count.checked_add(segment_result_bindings.len()))
                        .ok_or_else(|| {
                            SubmissionWaveDispatchError::Contract(invalid_operation(
                                "reusable execution command count overflows usize",
                            ))
                        })?;
                    encoded_provider_command_count = encoded_provider_command_count
                        .checked_add(operation_command_count)
                        .ok_or_else(|| {
                            SubmissionWaveDispatchError::Contract(invalid_operation(
                                "submission wave command count overflows usize",
                            ))
                        })?;
                    encoded_operations.push((
                        segment.start_node_index(),
                        segment_dynamic_bindings,
                        compute,
                        segment_result_bindings,
                    ));
                    node_index = segment.end_node_index() as usize;
                    segment_index += 1;
                    continue;
                }
                if reusable_program
                    .segments()
                    .get(segment_index)
                    .is_some_and(|segment| (segment.start_node_index() as usize) < node_index)
                {
                    return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                        "reusable execution segment traversal is not canonical",
                    )));
                }
                let provider = &providers[node_index];
                let node_identity = batch_identity
                    .materialize_node(node_index)
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                let invocation = BatchedOperationInvocation::from_wave_node(
                    runtime,
                    resolved,
                    provider.dispatch(),
                    batch_identity,
                    node_identity,
                    completion.wave(),
                    node_index,
                    active_bindings.clone(),
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
                let expected_phase = invocation.operation().profile_phase;
                let program_binding = invocation.program_binding().cloned();
                let operation = match provider.provider().encode_selected(invocation) {
                    Ok(operation) => operation,
                    Err(failure)
                        if node_identity.contains_identity(failure.identity())
                            && failure.phase() == expected_phase =>
                    {
                        return Err(SubmissionWaveDispatchError::Provider(failure));
                    }
                    Err(_) => {
                        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                            "wave provider returned a failure for another node identity or profile phase",
                        )));
                    }
                };
                let program_binding_count = operation.program_binding_count();
                validate_program_binding_patch(
                    resolved,
                    node_identity,
                    program_binding.as_ref(),
                    program_binding_count,
                    &mut program_binding_resources,
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
                let operation_command_count = operation
                    .dynamic_binding_count()
                    .checked_add(1)
                    .and_then(|count| count.checked_add(operation.result_binding_count()))
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "provider operation command count overflows usize",
                        ))
                    })?;
                encoded_provider_command_count = encoded_provider_command_count
                    .checked_add(operation_command_count)
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission wave command count overflows usize",
                        ))
                    })?;
                let encoded_node_index = u32::try_from(node_index).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission wave node index exceeds u32",
                    ))
                })?;
                let (mut node_program_bindings, dynamic_bindings, compute, result_bindings) =
                    operation.into_parts();
                program_bindings.append(&mut node_program_bindings);
                encoded_operations.push((
                    encoded_node_index,
                    dynamic_bindings,
                    compute,
                    result_bindings,
                ));
                node_index += 1;
            }
            if segment_index != reusable_program.segments().len() {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "reusable execution program contains an unreachable segment",
                )));
            }
        } else {
            for (node_index, (provider, node_identity)) in
                providers.iter().zip(batch_identity.nodes()).enumerate()
            {
                let invocation = BatchedOperationInvocation::from_wave_node(
                    runtime,
                    resolved,
                    provider.dispatch(),
                    batch_identity,
                    node_identity,
                    completion.wave(),
                    node_index,
                    active_bindings.clone(),
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
                let expected_phase = invocation.operation().profile_phase;
                let program_binding = invocation.program_binding().cloned();
                let operation = match provider.provider().encode_selected(invocation) {
                    Ok(operation) => operation,
                    Err(failure)
                        if node_identity.contains_identity(failure.identity())
                            && failure.phase() == expected_phase =>
                    {
                        return Err(SubmissionWaveDispatchError::Provider(failure));
                    }
                    Err(_) => {
                        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                        "wave provider returned a failure for another node identity or profile phase",
                    )));
                    }
                };
                let program_binding_count = operation.program_binding_count();
                validate_program_binding_patch(
                    resolved,
                    node_identity,
                    program_binding.as_ref(),
                    program_binding_count,
                    &mut program_binding_resources,
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
                let operation_command_count = operation
                    .dynamic_binding_count()
                    .checked_add(1)
                    .and_then(|count| count.checked_add(operation.result_binding_count()))
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "provider operation command count overflows usize",
                        ))
                    })?;
                let has_per_wave_bindings = program_binding_count > 0
                    || operation.dynamic_binding_count() > 0
                    || operation.result_binding_count() > 0;
                encoded_provider_command_count = encoded_provider_command_count
                    .checked_add(operation_command_count)
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission wave command count overflows usize",
                        ))
                    })?;
                let node_index = u32::try_from(node_index).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission wave node index exceeds u32",
                    ))
                })?;
                if has_per_wave_bindings {
                    reusable_execution_binding_nodes.push(node_index);
                }
                let (mut node_program_bindings, dynamic_bindings, compute, result_bindings) =
                    operation.into_parts();
                program_bindings.append(&mut node_program_bindings);
                encoded_operations.push((node_index, dynamic_bindings, compute, result_bindings));
            }
        }
        if let Some(layout) = completion.wave().claimed_backing().program_binding_layout() {
            let selected_plan_nodes = completion
                .wave()
                .nodes()
                .iter()
                .map(|node| node.plan_node_index())
                .collect::<BTreeSet<_>>();
            let expected_resources = layout
                .slots()
                .iter()
                .filter(|slot| selected_plan_nodes.contains(&slot.node_index()))
                .map(|slot| slot.resource_id())
                .collect::<BTreeSet<_>>();
            if program_binding_resources.len() != expected_resources.len()
                || expected_resources
                    .iter()
                    .any(|resource| !program_binding_resources.contains(*resource))
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "provider program binding patches do not cover the selected compiled layout exactly",
                )));
            }
        }
        let uncoalesced_program_binding_count = program_bindings.len();
        let coalesced_program_bindings = runtime
            .coalesce_program_bindings(program_bindings)
            .map_err(|error| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "device runtime could not coalesce program bindings: {error}"
                )))
            })?;
        if (uncoalesced_program_binding_count == 0) != coalesced_program_bindings.is_empty()
            || coalesced_program_bindings.len() > uncoalesced_program_binding_count
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "device runtime changed the program binding boundary cardinality illegally",
            )));
        }
        encoded_provider_command_count = encoded_provider_command_count
            .checked_add(coalesced_program_bindings.len())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "coalesced program binding command count overflows usize",
                ))
            })?;
        for command in coalesced_program_bindings {
            commands.push_dynamic_binding(command);
        }
        for (node_index, dynamic_bindings, compute, result_bindings) in encoded_operations {
            commands.push_operation_parts(node_index, dynamic_bindings, compute, result_bindings);
        }
        if reusable_program.is_none() {
            if let Some(program_id) = reusable_execution_program_id {
                commands
                    .set_reusable_execution_capture(DeviceReusableExecutionCapture::new(
                        program_id,
                        reusable_execution_binding_nodes,
                    ))
                    .map_err(SubmissionWaveDispatchError::Contract)?;
            }
        }
        if commands.len()
            != pre_provider_command_count.saturating_add(encoded_provider_command_count)
            || !lane.current_descriptor_matches_snapshot()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave encode command phases differ from provider-declared operation boundaries",
            )));
        }
        drop(provider_stage);

        let lane_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::LaneReserveSubmitAndArm,
        );
        let lane_reserve_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::LaneReserve,
        );
        let mut lane_reservation = lane
            .reserve_enqueue()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        drop(lane_reserve_stage);

        let device_submit_stage = SubmissionWaveDispatchStageTimer::start(
            timing_sink,
            SubmissionWaveDispatchStage::DeviceRuntimeSubmit,
        );
        completion.mark_submission_started();
        let submit_outcome = lane_reservation.submit_with_timing(commands, timing_sink);
        drop(lane_reservation);
        drop(device_submit_stage);

        let outcome = match submit_outcome {
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                let retry = completion
                    .definitely_not_submitted_wave()
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                let failures = batch_identity
                    .participants()
                    .iter()
                    .map(|participant| {
                        classify_device_error(runtime, participant.identity().clone(), &error)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { failures, retry })
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                let recovery = completion.submission_indeterminate();
                Err(SubmissionWaveDispatchError::SubmissionIndeterminate { recovery })
            }
            LaneSubmitOutcome::Submitted(fence) => {
                let device_attribution = runtime.submission_attribution(&fence);
                let completion_arm_stage = SubmissionWaveDispatchStageTimer::start(
                    timing_sink,
                    SubmissionWaveDispatchStage::CompletionArm,
                );
                let completion = match completion.arm(fence, timing_mode) {
                    Ok(completion) => completion,
                    Err((error, completion)) => {
                        return Err(SubmissionWaveDispatchError::PostSubmitContract {
                            error,
                            completion,
                        });
                    }
                };
                if !lane.current_descriptor_matches_snapshot() {
                    lane.fail_closed();
                    return Err(SubmissionWaveDispatchError::PostSubmitContract {
                        error: invalid_operation("wave submit completion runtime drifted"),
                        completion,
                    });
                }
                let attribution = match device_attribution
                    .map(|device| {
                        BoundDeviceSubmissionAttribution::new(
                            batch_identity.clone(),
                            completion.receipt().fingerprint().to_owned(),
                            device,
                        )
                    })
                    .transpose()
                {
                    Ok(attribution) => attribution,
                    Err(error) => {
                        return Err(SubmissionWaveDispatchError::PostSubmitContract {
                            error,
                            completion,
                        });
                    }
                };
                drop(completion_arm_stage);
                Ok(ProfiledSubmissionHandle {
                    completion,
                    attribution,
                })
            }
        };
        drop(lane_stage);
        outcome
    }
}

fn map_backing_initialization_error<R, Retry>(
    runtime: &R,
    batch_identity: &BatchOperationIdentity,
    error: BackingInitializationEncodeError<R::Error>,
) -> OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    match error {
        BackingInitializationEncodeError::Contract(error) => {
            OperationDispatchError::Contract(error)
        }
        BackingInitializationEncodeError::Runtime { participant, error } => {
            let identity = batch_identity
                .nodes()
                .iter()
                .flat_map(BatchOperationNodeIdentity::participants)
                .find(|candidate| {
                    candidate.node_key().sequence_authority() == participant.sequence_authority()
                        && candidate.node_key().request_authority()
                            == participant.request_authority()
                })
                .map(BatchOperationParticipantIdentity::identity)
                .cloned();
            let Some(identity) = identity else {
                return OperationDispatchError::Contract(invalid_operation(
                    "backing initialization failure has no matching batch participant",
                ));
            };
            match classify_device_error(runtime, identity, &error) {
                Ok(failure) => OperationDispatchError::Initialization(failure),
                Err(error) => OperationDispatchError::Contract(error),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_submission_wave_backing_upload<R>(
    runtime: &R,
    identity: &ExecutionIdentityEnvelope,
    backing: &LogicalBackingBufferView<'_, R::Buffer>,
    expected_usage: BufferUsage,
    element_type: ElementType,
    logical_offset_bytes: u64,
    bytes: &[u8],
    context: &'static str,
    mut push: impl FnMut(R::Command),
) -> Result<usize, SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
{
    let byte_len = u64::try_from(bytes.len()).map_err(|_| {
        SubmissionWaveDispatchError::Contract(invalid_operation(format!(
            "{context} byte length exceeds u64"
        )))
    })?;
    let element_bytes = element_type.size_bytes();
    let destination_end = logical_offset_bytes.checked_add(byte_len).ok_or_else(|| {
        SubmissionWaveDispatchError::Contract(invalid_operation(format!(
            "{context} destination range overflows"
        )))
    })?;
    if byte_len == 0
        || byte_len % element_bytes != 0
        || backing.usage() != expected_usage
        || backing.element_type() != element_type
        || destination_end > backing.size_bytes()
    {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            format!("{context} differs from its resolved logical backing"),
        )));
    }

    let mut logical_cursor = 0_u64;
    let mut encoded_bytes = 0_u64;
    let mut command_count = 0_usize;
    for segment in backing.segment_bindings() {
        let segment_end = logical_cursor
            .checked_add(segment.segment().length_bytes())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} backing coverage overflows"
                )))
            })?;
        let overlap_start = logical_cursor.max(logical_offset_bytes);
        let overlap_end = segment_end.min(destination_end);
        if overlap_start < overlap_end {
            let source_start =
                usize::try_from(overlap_start - logical_offset_bytes).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} source offset exceeds host address space"
                    )))
                })?;
            let piece_bytes = overlap_end - overlap_start;
            let source_end = source_start
                .checked_add(usize::try_from(piece_bytes).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} piece exceeds host address space"
                    )))
                })?)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} source range overflows"
                    )))
                })?;
            let destination_offset = segment
                .segment()
                .offset_bytes()
                .checked_add(overlap_start - logical_cursor)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} physical offset overflows"
                    )))
                })?;
            if piece_bytes % element_bytes != 0
                || destination_offset % element_bytes != 0
                || source_end > bytes.len()
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    format!("{context} splits an element or exceeds its source"),
                )));
            }
            let actual = runtime.buffer_descriptor(segment.buffer());
            if &actual != segment.descriptor()
                || destination_offset
                    .checked_add(piece_bytes)
                    .is_none_or(|end| end > actual.size_bytes)
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    format!("{context} backing descriptor drifted"),
                )));
            }
            let layout = HostTransferLayout::new(element_type, piece_bytes / element_bytes)
                .map_err(SubmissionWaveDispatchError::Contract)?;
            let command = runtime
                .encode_upload(
                    &bytes[source_start..source_end],
                    layout,
                    segment.buffer(),
                    destination_offset,
                )
                .map_err(|error| {
                    classify_device_error(runtime, identity.clone(), &error)
                        .map(SubmissionWaveDispatchError::InputUpload)
                        .unwrap_or_else(SubmissionWaveDispatchError::Contract)
                })?;
            push(command);
            command_count = command_count.checked_add(1).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} command count overflows usize"
                )))
            })?;
            encoded_bytes = encoded_bytes.checked_add(piece_bytes).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} encoded byte count overflows"
                )))
            })?;
        }
        logical_cursor = segment_end;
    }
    if encoded_bytes != byte_len {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            format!("{context} backing does not cover its complete range"),
        )));
    }
    Ok(command_count)
}

fn encode_submission_wave_determinism_restore<R>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    batch_identity: &BatchOperationIdentity,
    completion: &CompletionReservation<R>,
    restore: &SubmissionWaveDeterminismRestore,
    commands: &mut DeviceCommandBatch<R::Command>,
) -> Result<usize, SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
{
    restore
        .validate_for(resolved)
        .map_err(SubmissionWaveDispatchError::Contract)?;
    let participant_count = usize::try_from(restore.participant_count()).map_err(|_| {
        SubmissionWaveDispatchError::Contract(invalid_operation(
            "determinism restore participant count exceeds host address space",
        ))
    })?;
    let mut command_count = 0_usize;
    for (initialization_index, initialization) in restore.initializations().iter().enumerate() {
        let location = initialization.location();
        if !initialization
            .consumer_node_ids()
            .iter()
            .any(|node_id| node_id == location.node_id())
            || initialization.consumer_node_ids().iter().any(|node_id| {
                batch_identity
                    .node_index(node_id)
                    .and_then(|index| batch_identity.node_participant_count(index))
                    != Some(participant_count)
            })
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "determinism restore consumer coverage differs from its submitted wave",
            )));
        }
        let node_index = batch_identity
            .node_index(location.node_id())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "determinism restore anchor node is absent from its submitted wave",
                ))
            })?;
        let node_index_u32 = u32::try_from(node_index).map_err(|_| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "determinism restore node index exceeds u32",
            ))
        })?;
        let node_identity = batch_identity
            .materialize_node(node_index)
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let prepared_node = completion.wave().nodes().get(node_index).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "determinism restore anchor node is absent from its prepared wave",
            ))
        })?;
        for participant_index in 0..participant_count {
            let participant_index_u32 = u32::try_from(participant_index).map_err(|_| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "determinism restore participant index exceeds u32",
                ))
            })?;
            let participant = node_identity
                .participants()
                .get(participant_index)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "determinism restore participant is absent from its anchor node",
                    ))
                })?;
            let participant_work = prepared_node
                .work_shape()
                .participant_work()
                .get(participant_index)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "determinism restore participant lacks exact token work",
                    ))
                })?;
            let logical_work = DeviceCommandLogicalWork::new(
                DeviceBatchingForm::Scalar,
                1,
                participant_work.token_span().immediate_tokens(),
            )
            .map_err(SubmissionWaveDispatchError::Contract)?;
            let bytes = restore
                .participant_payloads(participant_index_u32)
                .and_then(|payloads| payloads.get(initialization_index))
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "determinism restore payload matrix is incomplete",
                    ))
                })?;
            let range = restore
                .initialization_range(participant_index_u32, initialization_index)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "determinism restore range matrix is incomplete",
                    ))
                })?;
            let backing = completion
                .backing_view(
                    location.node_id(),
                    participant_index_u32,
                    location.resource_id(),
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
            let encoded = encode_submission_wave_backing_upload(
                runtime,
                participant.identity(),
                &backing,
                location.usage(),
                location.element_type(),
                range.logical_offset_bytes(),
                bytes,
                "determinism restore",
                |command| {
                    commands.push_node_initialization(node_index_u32, logical_work, command);
                },
            )?;
            command_count = command_count.checked_add(encoded).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "determinism restore command count overflows usize",
                ))
            })?;
        }
    }
    Ok(command_count)
}

fn encode_submission_wave_inputs<R>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    batch_identity: &BatchOperationIdentity,
    completion: &CompletionReservation<R>,
    uploads: &[SubmissionWaveInputUpload],
    commands: &mut DeviceCommandBatch<R::Command>,
) -> Result<(), SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
{
    for upload in uploads {
        let node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == upload.node_id())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload references an unknown plan node",
                ))
            })?;
        let identity_node_index = batch_identity.node_index(upload.node_id()).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload has no physical node identity",
            ))
        })?;
        let node_identity = batch_identity
            .materialize_node(identity_node_index)
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let participant = node_identity
            .participants()
            .get(usize::try_from(upload.participant_index()).map_err(|_| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload participant index exceeds host address space",
                ))
            })?)
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload participant is absent from its plan node",
                ))
            })?;
        let value = node
            .values()
            .iter()
            .find(|value| {
                value.role() == ResolvedValueRole::Input
                    && value.ordinal() == upload.input_ordinal()
            })
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload references an unknown node input",
                ))
            })?;
        let [component] = value.storage().components() else {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload requires one activation storage component",
            )));
        };
        let byte_len = upload.source_layout().byte_len().map_err(|error| {
            SubmissionWaveDispatchError::Contract(invalid_operation(error.to_string()))
        })?;
        let value_end = upload
            .logical_offset_bytes()
            .checked_add(byte_len)
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload value range overflows",
                ))
            })?;
        if value.usage() != BufferUsage::Activations
            || !matches!(value.access(), TensorAccess::Read | TensorAccess::ReadWrite)
            || value.tensor().element_type() != upload.source_layout().element_type()
            || component.element_type() != upload.source_layout().element_type()
            || value_end > component.length_bytes()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload differs from its resolved activation binding",
            )));
        }
        let destination_start = component
            .offset_bytes()
            .checked_add(upload.logical_offset_bytes())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "submission input upload destination overflows",
                ))
            })?;
        let destination_end = destination_start.checked_add(byte_len).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload destination range overflows",
            ))
        })?;
        let backing = completion
            .backing_view(
                upload.node_id(),
                upload.participant_index(),
                component.resource_id(),
            )
            .map_err(SubmissionWaveDispatchError::Contract)?;
        if backing.usage() != BufferUsage::Activations
            || backing.element_type() != upload.source_layout().element_type()
            || destination_end > backing.size_bytes()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload backing differs from its resolved activation",
            )));
        }

        let element_bytes = upload.source_layout().element_type().size_bytes();
        let mut logical_cursor = 0_u64;
        let mut encoded_bytes = 0_u64;
        for segment in backing.segment_bindings() {
            let segment_end = logical_cursor
                .checked_add(segment.segment().length_bytes())
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission input upload backing coverage overflows",
                    ))
                })?;
            let overlap_start = logical_cursor.max(destination_start);
            let overlap_end = segment_end.min(destination_end);
            if overlap_start < overlap_end {
                let source_start =
                    usize::try_from(overlap_start - destination_start).map_err(|_| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission input upload source offset exceeds host address space",
                        ))
                    })?;
                let piece_bytes = overlap_end - overlap_start;
                let source_end = source_start
                    .checked_add(usize::try_from(piece_bytes).map_err(|_| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission input upload piece exceeds host address space",
                        ))
                    })?)
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission input upload source range overflows",
                        ))
                    })?;
                let destination_offset = segment
                    .segment()
                    .offset_bytes()
                    .checked_add(overlap_start - logical_cursor)
                    .ok_or_else(|| {
                        SubmissionWaveDispatchError::Contract(invalid_operation(
                            "submission input upload physical offset overflows",
                        ))
                    })?;
                if piece_bytes % element_bytes != 0
                    || destination_offset % element_bytes != 0
                    || source_end > upload.bytes().len()
                {
                    return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission input upload splits an element or exceeds its source",
                    )));
                }
                let actual = runtime.buffer_descriptor(segment.buffer());
                if &actual != segment.descriptor()
                    || destination_offset
                        .checked_add(piece_bytes)
                        .is_none_or(|end| end > actual.size_bytes)
                {
                    return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission input upload backing descriptor drifted",
                    )));
                }
                let layout = HostTransferLayout::new(
                    upload.source_layout().element_type(),
                    piece_bytes / element_bytes,
                )
                .map_err(SubmissionWaveDispatchError::Contract)?;
                let command = runtime
                    .encode_upload(
                        &upload.bytes()[source_start..source_end],
                        layout,
                        segment.buffer(),
                        destination_offset,
                    )
                    .map_err(|error| {
                        classify_device_error(runtime, participant.identity().clone(), &error)
                            .map(SubmissionWaveDispatchError::InputUpload)
                            .unwrap_or_else(SubmissionWaveDispatchError::Contract)
                    })?;
                commands.push_dynamic_binding(command);
                encoded_bytes = encoded_bytes.checked_add(piece_bytes).ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(
                        "submission input upload encoded byte count overflows",
                    ))
                })?;
            }
            logical_cursor = segment_end;
        }
        if encoded_bytes != byte_len {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "submission input upload backing does not cover its complete range",
            )));
        }
    }
    Ok(())
}
