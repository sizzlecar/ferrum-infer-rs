//! CUDA implementation of the vNext device ownership boundary.
//!
//! This module deliberately owns byte-addressed allocations instead of
//! adapting the legacy `Backend::Buffer`. vNext plans describe physical byte
//! regions, and operation providers must enqueue work only after core grants
//! submission authority.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use cudarc::cublas::{result::CublasError, CudaBlas};
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DriverError};
use ferrum_interfaces::vnext::{
    BufferDescriptor, CapabilityId, CopyRegion, DefinitelyNotSubmitted, DeviceBatchingForm,
    DeviceBufferRetention, DeviceClass, DeviceCommandBatch, DeviceCommandEntry,
    DeviceCommandExecutionTiming, DeviceDescriptor, DeviceErrorReport, DeviceExecutionInterval,
    DeviceExecutionIntervalKind, DeviceExecutionPath, DeviceExecutionTiming, DeviceId,
    DeviceNativeWorkAttribution, DeviceReusableAddressScope, DeviceReusableExecutionObservation,
    DeviceReusableExecutionPlan, DeviceReusableExecutionPreparation, DeviceReusableExecutionTrim,
    DeviceRuntime, DeviceSubmissionAttribution, DeviceSubmissionExecutionTiming,
    DeviceSubmissionStage, DeviceSubmissionTimingSink, DeviceTerminal, DeviceTerminalReceipt,
    DeviceTimingMeasurement, DeviceTimingMode, DeviceTimingUnavailableReason,
    DisabledDeviceSubmissionTimingSink, DynamicStorageProfile, ElementType, FenceIndeterminate,
    FenceQuery, HostTransferLayout, StreamState, VNextError,
};

use super::vnext_replay::{cuda_executable_candidates, CudaCommandReplayKey, CudaExecutableCache};

static NEXT_RUNTIME_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_STREAM_INSTANCE: AtomicU64 = AtomicU64::new(1);

struct CudaSubmissionStageTimer<'sink, S>
where
    S: DeviceSubmissionTimingSink,
{
    sink: &'sink S,
    stage: DeviceSubmissionStage,
    started: Option<Instant>,
}

impl<'sink, S> CudaSubmissionStageTimer<'sink, S>
where
    S: DeviceSubmissionTimingSink,
{
    #[inline(always)]
    fn start(sink: &'sink S, stage: DeviceSubmissionStage) -> Self {
        Self {
            sink,
            stage,
            started: S::ENABLED.then(Instant::now),
        }
    }
}

impl<S> Drop for CudaSubmissionStageTimer<'_, S>
where
    S: DeviceSubmissionTimingSink,
{
    fn drop(&mut self) {
        if let Some(started) = self.started.take() {
            if !std::thread::panicking() {
                self.sink
                    .record_device_submission(self.stage, started.elapsed());
            }
        }
    }
}

/// Typed construction input supplied by the CUDA composition root.
///
/// Capability and storage profiles come from the installed provider bundle;
/// the device runtime does not infer them from a model, GPU name, or memory
/// size. The implementation fingerprint must identify that exact bundle.
pub struct CudaDeviceRuntimeConfig {
    pub ordinal: usize,
    pub device_id: DeviceId,
    pub runtime_implementation_fingerprint: String,
    pub capabilities: BTreeSet<CapabilityId>,
    pub dynamic_storage_profiles: BTreeSet<DynamicStorageProfile>,
}

#[derive(Debug)]
pub enum CudaDeviceRuntimeError {
    Contract(String),
    Driver {
        operation: &'static str,
        source: DriverError,
    },
    Blas {
        operation: &'static str,
        source: CublasError,
    },
}

impl CudaDeviceRuntimeError {
    pub(super) fn contract(message: impl Into<String>) -> Self {
        Self::Contract(message.into())
    }

    pub(super) fn driver(operation: &'static str, source: DriverError) -> Self {
        Self::Driver { operation, source }
    }

    pub(super) fn blas(operation: &'static str, source: CublasError) -> Self {
        Self::Blas { operation, source }
    }

    fn driver_code(&self) -> Option<cudarc::driver::sys::CUresult> {
        match self {
            Self::Contract(_) => None,
            Self::Driver { source, .. } => Some(source.0),
            Self::Blas { .. } => None,
        }
    }
}

impl fmt::Display for CudaDeviceRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(message) => formatter.write_str(message),
            Self::Driver { operation, source } => {
                write!(formatter, "CUDA {operation} failed: {source:?}")
            }
            Self::Blas { operation, source } => {
                write!(formatter, "CUDA {operation} failed: {source:?}")
            }
        }
    }
}

impl Error for CudaDeviceRuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contract(_) => None,
            Self::Driver { .. } => None,
            Self::Blas { source, .. } => Some(source),
        }
    }
}

struct CudaAllocation {
    _base: CudaSlice<u8>,
    aligned_ptr: cudarc::driver::sys::CUdeviceptr,
    requested_bytes: u64,
}

unsafe impl Send for CudaAllocation {}
unsafe impl Sync for CudaAllocation {}

/// One core-owned CUDA allocation with its exact admitted descriptor.
pub struct CudaDeviceBuffer {
    descriptor: BufferDescriptor,
    runtime_instance: u64,
    allocation: Arc<CudaAllocation>,
}

impl fmt::Debug for CudaDeviceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceBuffer")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl CudaDeviceBuffer {
    fn region(&self, range: Range<u64>) -> Result<CudaBufferRegion, CudaDeviceRuntimeError> {
        self.region_with_retention(range, None)
    }

    pub(crate) fn retained_region(
        &self,
        range: Range<u64>,
        retention: DeviceBufferRetention,
    ) -> Result<CudaBufferRegion, CudaDeviceRuntimeError> {
        self.region_with_retention(range, Some(retention))
    }

    fn region_with_retention(
        &self,
        range: Range<u64>,
        core_retention: Option<DeviceBufferRetention>,
    ) -> Result<CudaBufferRegion, CudaDeviceRuntimeError> {
        if range.start >= range.end
            || range.end > self.descriptor.size_bytes
            || range.end > self.allocation.requested_bytes
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA buffer region is empty or outside its admitted allocation",
            ));
        }
        let device_ptr = self
            .allocation
            .aligned_ptr
            .checked_add(range.start)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA buffer pointer overflow"))?;
        let reusable_address_scope = core_retention
            .as_ref()
            .and_then(DeviceBufferRetention::reusable_address_scope);
        Ok(CudaBufferRegion {
            _allocation: Arc::clone(&self.allocation),
            _core_retention: core_retention,
            reusable_address_scope,
            runtime_instance: self.runtime_instance,
            device_ptr,
            length_bytes: range.end - range.start,
            element_type: self.descriptor.element_type,
        })
    }
}

/// Owned physical CUDA range retained by an encoded command and its fence.
#[derive(Clone)]
pub(crate) struct CudaBufferRegion {
    _allocation: Arc<CudaAllocation>,
    _core_retention: Option<DeviceBufferRetention>,
    reusable_address_scope: Option<DeviceReusableAddressScope>,
    runtime_instance: u64,
    device_ptr: cudarc::driver::sys::CUdeviceptr,
    length_bytes: u64,
    element_type: ElementType,
}

impl CudaBufferRegion {
    pub(crate) const fn device_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.device_ptr
    }

    pub(crate) const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub(crate) const fn element_type(&self) -> ElementType {
        self.element_type
    }
}

type EnqueueAction = Box<
    dyn Fn(
            &CudaStream,
            &CudaBlas,
            &[CudaBufferRegion],
            &[Box<[u8]>],
        ) -> Result<(), CudaDeviceRuntimeError>
        + Send
        + 'static,
>;

/// CUDA work captured by a reusable executable. Submission-scoped resource
/// dependencies must stay on `CudaDeviceCommand` so a cached executable never
/// retains request- or sequence-owned allocations.
pub(crate) struct CudaCommandExecutable {
    regions: Vec<CudaBufferRegion>,
    host_storage: Vec<Box<[u8]>>,
    enqueue: Mutex<EnqueueAction>,
}

/// Encoded CUDA work. Buffer and host-transfer storage stays alive until the
/// returned fence reaches a terminal state.
pub struct CudaDeviceCommand {
    runtime_instance: u64,
    operation: &'static str,
    batching_form: DeviceBatchingForm,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    executable: Arc<CudaCommandExecutable>,
    fence_dependencies: Vec<CudaBufferRegion>,
    replay_key: Option<CudaCommandReplayKey>,
    reusable_address_scope: Option<DeviceReusableAddressScope>,
}

impl fmt::Debug for CudaDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceCommand")
            .field("runtime_instance", &self.runtime_instance)
            .field("operation", &self.operation)
            .field("batching_form", &self.batching_form)
            .field("participant_count", &self.participant_count)
            .field("token_count", &self.token_count)
            .field("compute_dispatch_count", &self.compute_dispatch_count)
            .field("transfer_command_count", &self.transfer_command_count)
            .field("captured_region_count", &self.executable.regions.len())
            .field(
                "captured_host_storage_count",
                &self.executable.host_storage.len(),
            )
            .field("fence_dependency_count", &self.fence_dependencies.len())
            .field("replayable", &self.replay_key.is_some())
            .finish_non_exhaustive()
    }
}

impl CudaDeviceCommand {
    /// Backend-local operation providers use this constructor after translating
    /// every logical invocation view into owned physical regions.
    pub(crate) fn operation(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        enqueue: impl Fn(&CudaStream, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            Vec::new(),
            None,
            move |stream, _blas, regions, _host_storage| enqueue(stream, regions),
        )
    }

    pub(crate) fn replayable_operation(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        replay_key: CudaCommandReplayKey,
        enqueue: impl Fn(&CudaStream, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            Vec::new(),
            Some(replay_key),
            move |stream, _blas, regions, _host_storage| enqueue(stream, regions),
        )
    }

    pub(crate) fn operation_with_blas(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        enqueue: impl Fn(&CudaStream, &CudaBlas, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            Vec::new(),
            None,
            move |stream, blas, regions, _host_storage| enqueue(stream, blas, regions),
        )
    }

    pub(crate) fn replayable_operation_with_blas(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        replay_key: CudaCommandReplayKey,
        enqueue: impl Fn(&CudaStream, &CudaBlas, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            Vec::new(),
            Some(replay_key),
            move |stream, blas, regions, _host_storage| enqueue(stream, blas, regions),
        )
    }

    /// Encodes replayable work whose launch addresses are stable while keeping
    /// additional submission-scoped allocations alive through the completion
    /// fence. Fence dependencies do not participate in graph identity or scope.
    pub(crate) fn replayable_operation_with_blas_and_fence_dependencies(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        fence_dependencies: Vec<CudaBufferRegion>,
        replay_key: CudaCommandReplayKey,
        enqueue: impl Fn(&CudaStream, &CudaBlas, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            fence_dependencies,
            Some(replay_key),
            move |stream, blas, regions, _host_storage| enqueue(stream, blas, regions),
        )
    }

    pub(crate) fn operation_with_host_storage_and_blas(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        enqueue: impl Fn(
                &CudaStream,
                &CudaBlas,
                &[CudaBufferRegion],
                &[Box<[u8]>],
            ) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_with_host_storage_and_blas_inner(
            operation,
            regions,
            host_storage,
            Vec::new(),
            None,
            enqueue,
        )
    }

    pub(crate) fn replayable_operation_with_host_storage_and_blas(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        replay_key: CudaCommandReplayKey,
        enqueue: impl Fn(
                &CudaStream,
                &CudaBlas,
                &[CudaBufferRegion],
                &[Box<[u8]>],
            ) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_with_host_storage_and_blas_inner(
            operation,
            regions,
            host_storage,
            Vec::new(),
            Some(replay_key),
            enqueue,
        )
    }

    fn operation_with_host_storage_and_blas_inner(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        fence_dependencies: Vec<CudaBufferRegion>,
        replay_key: Option<CudaCommandReplayKey>,
        enqueue: impl Fn(
                &CudaStream,
                &CudaBlas,
                &[CudaBufferRegion],
                &[Box<[u8]>],
            ) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let runtime_instance = common_runtime_instance(&regions)?;
        validate_fence_dependencies(runtime_instance, &fence_dependencies)?;
        if host_storage.iter().any(|storage| storage.is_empty()) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA operation host storage contains an empty region",
            ));
        }
        let (replay_key, reusable_address_scope) =
            bind_replay_contract(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: Arc::new(CudaCommandExecutable {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            }),
            fence_dependencies,
            replay_key,
            reusable_address_scope,
        })
    }

    fn operation_inner(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        fence_dependencies: Vec<CudaBufferRegion>,
        replay_key: Option<CudaCommandReplayKey>,
        enqueue: impl Fn(
                &CudaStream,
                &CudaBlas,
                &[CudaBufferRegion],
                &[Box<[u8]>],
            ) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let runtime_instance = common_runtime_instance(&regions)?;
        validate_fence_dependencies(runtime_instance, &fence_dependencies)?;
        let host_storage = Vec::new();
        let (replay_key, reusable_address_scope) =
            bind_replay_contract(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: Arc::new(CudaCommandExecutable {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            }),
            fence_dependencies,
            replay_key,
            reusable_address_scope,
        })
    }

    fn transfer(
        runtime_instance: u64,
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        enqueue: EnqueueAction,
    ) -> Self {
        let executable = Arc::new(CudaCommandExecutable {
            regions,
            host_storage,
            enqueue: Mutex::new(enqueue),
        });
        Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            executable,
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
        }
    }

    pub(crate) fn with_work_attribution(
        mut self,
        batching_form: DeviceBatchingForm,
        participant_count: u32,
        token_count: u64,
        compute_dispatch_count: u64,
        transfer_command_count: u64,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if participant_count == 0 || (compute_dispatch_count == 0 && transfer_command_count == 0) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA operation attribution has no participants or native work",
            ));
        }
        self.batching_form = batching_form;
        self.participant_count = participant_count;
        self.token_count = token_count;
        self.compute_dispatch_count = compute_dispatch_count;
        self.transfer_command_count = transfer_command_count;
        Ok(self)
    }

    pub(crate) fn enqueue(
        &self,
        stream: &CudaStream,
        blas: &CudaBlas,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let enqueue = self
            .executable
            .enqueue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enqueue(
            stream,
            blas,
            &self.executable.regions,
            &self.executable.host_storage,
        )
    }

    pub(crate) const fn replay_key(&self) -> Option<CudaCommandReplayKey> {
        self.replay_key
    }

    pub(crate) const fn reusable_address_scope(&self) -> Option<DeviceReusableAddressScope> {
        self.reusable_address_scope
    }

    pub(crate) fn executable(&self) -> Arc<CudaCommandExecutable> {
        Arc::clone(&self.executable)
    }
}

fn bind_replay_contract(
    replay_key: Option<CudaCommandReplayKey>,
    operation: &'static str,
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> (
    Option<CudaCommandReplayKey>,
    Option<DeviceReusableAddressScope>,
) {
    let Some(key) = replay_key else {
        return (None, None);
    };
    let mut scope = DeviceReusableAddressScope::Plan;
    for region in regions {
        let Some(region_scope) = region.reusable_address_scope else {
            return (None, None);
        };
        match region_scope {
            DeviceReusableAddressScope::Plan => {}
            DeviceReusableAddressScope::ExecutionLane(lane_id) => match scope {
                DeviceReusableAddressScope::Plan => {
                    scope = DeviceReusableAddressScope::ExecutionLane(lane_id);
                }
                DeviceReusableAddressScope::ExecutionLane(current) if current == lane_id => {}
                DeviceReusableAddressScope::ExecutionLane(_) => return (None, None),
            },
        }
    }
    (
        Some(
            key.bind_runtime_payload(
                operation,
                regions
                    .iter()
                    .map(|region| (region.device_ptr, region.length_bytes, region.element_type)),
                host_storage,
            ),
        ),
        Some(scope),
    )
}

fn common_runtime_instance(regions: &[CudaBufferRegion]) -> Result<u64, CudaDeviceRuntimeError> {
    let runtime_instance = regions
        .first()
        .map(|region| region.runtime_instance)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA operation has no buffer regions"))?;
    if regions
        .iter()
        .any(|region| region.runtime_instance != runtime_instance)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA operation mixes buffers from different runtime instances",
        ));
    }
    Ok(runtime_instance)
}

fn validate_fence_dependencies(
    runtime_instance: u64,
    dependencies: &[CudaBufferRegion],
) -> Result<(), CudaDeviceRuntimeError> {
    if dependencies
        .iter()
        .any(|region| region.runtime_instance != runtime_instance)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA operation retains fence dependencies from another runtime instance",
        ));
    }
    Ok(())
}

fn cuda_submission_attribution(
    command_phases: &[ferrum_interfaces::vnext::DeviceCommandPhase],
    command_node_indices: &[Option<u32>],
    commands: &[CudaDeviceCommand],
    execution_paths: &[DeviceExecutionPath],
) -> Result<DeviceSubmissionAttribution, CudaDeviceRuntimeError> {
    if command_phases.len() != commands.len()
        || command_node_indices.len() != commands.len()
        || execution_paths.len() != commands.len()
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA command attribution differs from its submitted batch",
        ));
    }
    let rows = commands
        .iter()
        .enumerate()
        .map(|(command_index, command)| {
            let command_index = u32::try_from(command_index)
                .map_err(|_| CudaDeviceRuntimeError::contract("CUDA command index exceeds u32"))?;
            DeviceNativeWorkAttribution::new(
                command_index,
                command_node_indices[command_index as usize],
                command_phases[command_index as usize],
                command.operation,
                execution_paths[command_index as usize],
                command.batching_form,
                command.participant_count,
                command.token_count,
                command.compute_dispatch_count,
                command.transfer_command_count,
            )
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA command attribution has invalid native work metadata",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    DeviceSubmissionAttribution::new(rows).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA submission attribution is empty or unordered")
    })
}

pub struct CudaDeviceStream {
    id: u64,
    runtime_instance: u64,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    state: Arc<CudaStreamState>,
    executable_cache: CudaExecutableCache,
}

impl fmt::Debug for CudaDeviceStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceStream")
            .field("id", &self.id)
            .field("runtime_instance", &self.runtime_instance)
            .field("state", &self.state.snapshot())
            .finish_non_exhaustive()
    }
}

impl Drop for CudaDeviceStream {
    fn drop(&mut self) {
        if !self.state.is_quiescent() {
            // An indeterminate lane must retain captured pointer ownership.
            // Normal executor shutdown reaches quiescence and destroys every
            // graph without a device-wide synchronization.
            self.executable_cache.leak_if_in_flight();
        }
    }
}

struct CudaStreamState {
    recording: AtomicBool,
    failed: AtomicBool,
    in_flight: AtomicU64,
}

impl CudaStreamState {
    fn new() -> Self {
        Self {
            recording: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            in_flight: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> StreamState {
        if self.failed.load(Ordering::Acquire) {
            StreamState::Failed
        } else if self.recording.load(Ordering::Acquire) {
            StreamState::Recording
        } else if self.in_flight.load(Ordering::Acquire) == 0 {
            StreamState::Ready
        } else {
            StreamState::Submitted
        }
    }

    fn is_quiescent(&self) -> bool {
        !self.failed.load(Ordering::Acquire)
            && !self.recording.load(Ordering::Acquire)
            && self.in_flight.load(Ordering::Acquire) == 0
    }

    fn begin_submission(&self) -> Result<(), CudaDeviceRuntimeError> {
        if self.failed.load(Ordering::Acquire)
            || self
                .recording
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA stream is failed or already recording a submission",
            ));
        }
        if self.failed.load(Ordering::Acquire) {
            self.recording.store(false, Ordering::Release);
            return Err(CudaDeviceRuntimeError::contract("CUDA stream is failed"));
        }
        Ok(())
    }

    fn submission_recorded(&self) -> Result<(), CudaDeviceRuntimeError> {
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA in-flight count overflowed"))?;
        self.recording.store(false, Ordering::Release);
        Ok(())
    }

    fn finish_one(&self) {
        let _ = self
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(1)
            });
    }

    fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        self.recording.store(false, Ordering::Release);
    }

    fn synchronized(&self) {
        self.in_flight.store(0, Ordering::Release);
        self.recording.store(false, Ordering::Release);
    }
}

pub struct CudaDeviceFence {
    event: CudaEvent,
    timing: CudaFenceTiming,
    command_timing: CudaFenceCommandTiming,
    attribution: Option<DeviceSubmissionAttribution>,
    stream_state: Arc<CudaStreamState>,
    terminal_accounted: AtomicBool,
    _stream: Arc<CudaStream>,
    _blas: Arc<CudaBlas>,
    _commands: Vec<CudaDeviceCommand>,
}

enum CudaFenceTiming {
    NotRequested,
    Events { start: CudaEvent },
    Unavailable,
}

struct CudaCommandEventTiming {
    command_index: u32,
    kind: DeviceExecutionIntervalKind,
    operation: &'static str,
    start: CudaEvent,
    end: CudaEvent,
}

enum CudaFenceCommandTiming {
    NotRequested,
    Unavailable(DeviceTimingUnavailableReason),
    Events(Vec<CudaCommandEventTiming>),
}

impl CudaFenceCommandTiming {
    fn measurement(&self) -> DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        match self {
            Self::NotRequested => DeviceTimingMeasurement::NotRequested,
            Self::Unavailable(reason) => DeviceTimingMeasurement::Unavailable(*reason),
            Self::Events(events) => {
                let Some(origin) = events.first().map(|event| &event.start) else {
                    return DeviceTimingMeasurement::Unavailable(
                        DeviceTimingUnavailableReason::BackendMeasurementFailed,
                    );
                };
                let commands = events
                    .iter()
                    .map(|event| {
                        let start_offset_ns = cuda_event_elapsed_ns(origin, &event.start)?;
                        let end_offset_ns = cuda_event_elapsed_ns(origin, &event.end)?;
                        let interval = DeviceExecutionInterval::new_labeled(
                            event.kind,
                            start_offset_ns,
                            end_offset_ns,
                            event.operation,
                        )?;
                        DeviceCommandExecutionTiming::new(event.command_index, vec![interval])
                    })
                    .collect::<Option<Vec<_>>>()
                    .and_then(DeviceSubmissionExecutionTiming::new);
                commands.map_or_else(
                    || {
                        DeviceTimingMeasurement::Unavailable(
                            DeviceTimingUnavailableReason::BackendMeasurementFailed,
                        )
                    },
                    DeviceTimingMeasurement::Measured,
                )
            }
        }
    }
}

impl fmt::Debug for CudaDeviceFence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceFence")
            .field("stream_state", &self.stream_state.snapshot())
            .finish_non_exhaustive()
    }
}

impl CudaDeviceFence {
    fn mark_terminal(&self) {
        if !self.terminal_accounted.swap(true, Ordering::AcqRel) {
            self.stream_state.finish_one();
        }
    }

    fn execution_timing(&self) -> DeviceTimingMeasurement<DeviceExecutionTiming> {
        let start = match &self.timing {
            CudaFenceTiming::Events { start } => start,
            _ => {
                return match &self.timing {
                    CudaFenceTiming::NotRequested => DeviceTimingMeasurement::NotRequested,
                    CudaFenceTiming::Unavailable => DeviceTimingMeasurement::Unavailable(
                        DeviceTimingUnavailableReason::BackendMeasurementFailed,
                    ),
                    CudaFenceTiming::Events { .. } => unreachable!(),
                };
            }
        };
        cuda_event_elapsed_ns(start, &self.event).map_or_else(
            || {
                DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                )
            },
            |elapsed_ns| {
                DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
                    elapsed_ns,
                ))
            },
        )
    }

    fn terminal_receipt<E>(&self, terminal: DeviceTerminal<E>) -> DeviceTerminalReceipt<E> {
        match &self.timing {
            CudaFenceTiming::NotRequested => DeviceTerminalReceipt::unprofiled(terminal),
            CudaFenceTiming::Events { .. } | CudaFenceTiming::Unavailable => {
                match &self.command_timing {
                    CudaFenceCommandTiming::NotRequested => {
                        DeviceTerminalReceipt::profiled(terminal, self.execution_timing())
                    }
                    CudaFenceCommandTiming::Unavailable(_) | CudaFenceCommandTiming::Events(_) => {
                        DeviceTerminalReceipt::profiled_with_submission_timing(
                            terminal,
                            self.execution_timing(),
                            self.command_timing.measurement(),
                        )
                    }
                }
            }
        }
    }
}

fn cuda_event_elapsed_ns(start: &CudaEvent, end: &CudaEvent) -> Option<u64> {
    let elapsed_ms =
        unsafe { cudarc::driver::result::event::elapsed(start.cu_event(), end.cu_event()) }.ok()?;
    if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
        return None;
    }
    let elapsed_ns = f64::from(elapsed_ms) * 1_000_000.0;
    (elapsed_ns <= u64::MAX as f64).then(|| elapsed_ns.round() as u64)
}

struct QuarantinedSubmission {
    stream_id: u64,
    _stream: Arc<CudaStream>,
    _blas: Arc<CudaBlas>,
    _commands: Vec<CudaDeviceCommand>,
}

/// Concrete CUDA primitive runtime consumed by the shared vNext resource and
/// operation dispatch layers.
pub struct CudaDeviceRuntime {
    descriptor: DeviceDescriptor,
    runtime_instance: u64,
    context: Arc<CudaContext>,
    allocation_stream: Arc<CudaStream>,
    quarantined: Mutex<Vec<QuarantinedSubmission>>,
}

impl fmt::Debug for CudaDeviceRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceRuntime")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl CudaDeviceRuntime {
    pub fn new(config: CudaDeviceRuntimeConfig) -> Result<Self, CudaDeviceRuntimeError> {
        let context = CudaContext::new(config.ordinal)
            .map_err(|error| CudaDeviceRuntimeError::driver("context creation", error))?;
        // vNext owns all cross-stream ordering through explicit commands and
        // fences. Per-slice implicit events would create a second authority.
        unsafe {
            context.disable_event_tracking();
        }
        let allocation_stream = context
            .new_stream()
            .map_err(|error| CudaDeviceRuntimeError::driver("allocation stream creation", error))?;
        let total_memory_bytes = u64::try_from(
            context
                .total_mem()
                .map_err(|error| CudaDeviceRuntimeError::driver("memory query", error))?,
        )
        .map_err(|_| CudaDeviceRuntimeError::contract("CUDA memory size exceeds u64"))?;
        let ordinal = u32::try_from(config.ordinal)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA ordinal exceeds u32"))?;
        let descriptor = DeviceDescriptor {
            id: config.device_id,
            class: DeviceClass::Accelerator,
            ordinal,
            total_memory_bytes,
            runtime_implementation_fingerprint: config.runtime_implementation_fingerprint,
            capabilities: config.capabilities,
            dynamic_storage_profiles: config.dynamic_storage_profiles,
        };
        descriptor
            .validate()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        let runtime_instance = NEXT_RUNTIME_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA runtime identity exhausted"))?;
        Ok(Self {
            descriptor,
            runtime_instance,
            context,
            allocation_stream,
            quarantined: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn validate_buffer(&self, buffer: &CudaDeviceBuffer) -> Result<(), CudaDeviceRuntimeError> {
        if buffer.runtime_instance != self.runtime_instance {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA buffer belongs to another runtime instance",
            ));
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &CudaDeviceStream) -> Result<(), CudaDeviceRuntimeError> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA stream belongs to another runtime instance",
            ));
        }
        Ok(())
    }

    fn quarantined(&self) -> MutexGuard<'_, Vec<QuarantinedSubmission>> {
        self.quarantined
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn quarantine(&self, stream: &CudaDeviceStream, commands: Vec<CudaDeviceCommand>) {
        self.quarantined().push(QuarantinedSubmission {
            stream_id: stream.id,
            _stream: Arc::clone(&stream.stream),
            _blas: Arc::clone(&stream.blas),
            _commands: commands,
        });
    }

    fn release_quarantine(&self, stream_id: u64) {
        self.quarantined()
            .retain(|submission| submission.stream_id != stream_id);
    }
}

fn checked_usize(value: u64, context: &'static str) -> Result<usize, CudaDeviceRuntimeError> {
    usize::try_from(value).map_err(|_| {
        CudaDeviceRuntimeError::contract(format!("{context} exceeds host address space"))
    })
}

fn checked_end(
    offset: u64,
    length: u64,
    capacity: u64,
    context: &'static str,
) -> Result<u64, CudaDeviceRuntimeError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CudaDeviceRuntimeError::contract(format!("{context} range overflows")))?;
    if length == 0 || end > capacity {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "{context} range is empty or outside its buffer"
        )));
    }
    Ok(end)
}

impl DeviceRuntime for CudaDeviceRuntime {
    type Buffer = CudaDeviceBuffer;
    type Stream = CudaDeviceStream;
    type Command = CudaDeviceCommand;
    type Fence = CudaDeviceFence;
    type Error = CudaDeviceRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(
        &self,
        permit: ferrum_interfaces::vnext::DeviceAllocationPermit<'_>,
    ) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        let extra_alignment = request
            .alignment_bytes()
            .checked_sub(1)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA allocation alignment is zero"))?;
        let allocation_bytes = request
            .size_bytes()
            .checked_add(extra_alignment)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA allocation size overflows"))?;
        let allocation_bytes = checked_usize(allocation_bytes, "CUDA allocation size")?;
        let base = unsafe { self.allocation_stream.alloc::<u8>(allocation_bytes) }
            .map_err(|error| CudaDeviceRuntimeError::driver("allocation", error))?;
        let (base_ptr, base_use) = base.device_ptr(&self.allocation_stream);
        drop(base_use);
        let alignment = request.alignment_bytes();
        let aligned_ptr = base_ptr
            .checked_add(alignment - 1)
            .map(|pointer| pointer & !(alignment - 1))
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA aligned pointer overflows"))?;
        self.allocation_stream
            .synchronize()
            .map_err(|error| CudaDeviceRuntimeError::driver("allocation synchronization", error))?;
        let descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        Ok(CudaDeviceBuffer {
            descriptor,
            runtime_instance: self.runtime_instance,
            allocation: Arc::new(CudaAllocation {
                _base: base,
                aligned_ptr,
                requested_bytes: request.size_bytes(),
            }),
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        let id = NEXT_STREAM_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA stream identity exhausted"))?;
        let stream = self
            .context
            .new_stream()
            .map_err(|error| CudaDeviceRuntimeError::driver("stream creation", error))?;
        let blas = Arc::new(
            CudaBlas::new(Arc::clone(&stream))
                .map_err(|error| CudaDeviceRuntimeError::blas("cuBLAS handle creation", error))?,
        );
        Ok(CudaDeviceStream {
            id,
            runtime_instance: self.runtime_instance,
            stream,
            blas,
            state: Arc::new(CudaStreamState::new()),
            executable_cache: CudaExecutableCache::new(),
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.runtime_instance != self.runtime_instance {
            return StreamState::Failed;
        }
        stream.state.snapshot()
    }

    fn configure_reusable_executables(
        &self,
        stream: &mut Self::Stream,
        plan: DeviceReusableExecutionPlan,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        self.validate_stream(stream)?;
        if !stream.state.is_quiescent() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable executable preparation requires its quiescent owning stream",
            ));
        }
        stream
            .executable_cache
            .configure(plan)
            .map_err(CudaDeviceRuntimeError::contract)
    }

    fn seal_reusable_executables(
        &self,
        stream: &mut Self::Stream,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        self.validate_stream(stream)?;
        if !stream.state.is_quiescent() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable executable sealing requires its quiescent owning stream",
            ));
        }
        stream
            .executable_cache
            .seal()
            .map_err(CudaDeviceRuntimeError::contract)
    }

    fn reusable_executable_preparation(
        &self,
        stream: &Self::Stream,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        self.validate_stream(stream)?;
        if !stream.state.is_quiescent() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable executable inspection requires its quiescent owning stream",
            ));
        }
        stream
            .executable_cache
            .preparation()
            .map_err(CudaDeviceRuntimeError::contract)
    }

    fn trim_reusable_executables(
        &self,
        stream: &mut Self::Stream,
    ) -> Result<DeviceReusableExecutionTrim, Self::Error> {
        if stream.runtime_instance != self.runtime_instance || !stream.state.is_quiescent() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable executable trim requires its quiescent owning stream",
            ));
        }
        let (released_executables, released_rejections) = stream.executable_cache.trim_quiescent();
        Ok(DeviceReusableExecutionTrim::new(
            released_executables,
            released_rejections,
        ))
    }

    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        self.validate_buffer(source)?;
        self.validate_buffer(destination)?;
        region
            .validate_bounds(&source.descriptor, &destination.descriptor)
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        if source.descriptor.element_type != destination.descriptor.element_type {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA copy requires matching source and destination element types",
            ));
        }
        let source_region = source.region(
            region.source_offset_bytes()..region.source_offset_bytes() + region.length_bytes(),
        )?;
        let destination_region = destination.region(
            region.destination_offset_bytes()
                ..region.destination_offset_bytes() + region.length_bytes(),
        )?;
        let regions = vec![source_region, destination_region];
        Ok(CudaDeviceCommand::transfer(
            self.runtime_instance,
            "device copy",
            regions,
            Vec::new(),
            Box::new(|stream, _blas, regions, _host_storage| {
                let bytes = checked_usize(regions[0].length_bytes, "CUDA copy length")?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        regions[1].device_ptr,
                        regions[0].device_ptr,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("device copy", error))
            }),
        ))
    }

    fn encode_upload(
        &self,
        source: &[u8],
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.validate_buffer(destination)?;
        source_layout
            .validate_bytes(source.len())
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        if source_layout.element_type() != destination.descriptor.element_type {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA upload layout differs from destination element type",
            ));
        }
        let source_bytes = u64::try_from(source.len())
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA upload size exceeds u64"))?;
        let destination_end = checked_end(
            destination_offset_bytes,
            source_bytes,
            destination.descriptor.size_bytes,
            "CUDA upload",
        )?;
        let destination_region = destination.region(destination_offset_bytes..destination_end)?;
        let host_storage = vec![source.to_vec().into_boxed_slice()];
        Ok(CudaDeviceCommand::transfer(
            self.runtime_instance,
            "host upload",
            vec![destination_region],
            host_storage,
            Box::new(|stream, _blas, regions, host_storage| {
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        regions[0].device_ptr,
                        host_storage[0].as_ref(),
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("host upload", error))
            }),
        ))
    }

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.validate_buffer(destination)?;
        let destination_end = checked_end(
            destination_offset_bytes,
            length_bytes,
            destination.descriptor.size_bytes,
            "CUDA zero",
        )?;
        let destination_region = destination.region(destination_offset_bytes..destination_end)?;
        Ok(CudaDeviceCommand::transfer(
            self.runtime_instance,
            "device zero",
            vec![destination_region],
            Vec::new(),
            Box::new(|stream, _blas, regions, _host_storage| {
                let bytes = checked_usize(regions[0].length_bytes, "CUDA zero length")?;
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        regions[0].device_ptr,
                        0,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("device zero", error))
            }),
        ))
    }

    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        self.submit_with_timing(stream, commands, &DisabledDeviceSubmissionTimingSink)
    }

    fn submit_with_timing<S>(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
        timing_sink: &S,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>>
    where
        S: DeviceSubmissionTimingSink,
    {
        let validate_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::ValidateAndPrepare);
        if let Err(error) = self.validate_stream(stream) {
            return Err(DefinitelyNotSubmitted::new(error));
        }
        if commands.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract("CUDA command batch is empty"),
            ));
        }
        let timing_mode = commands.timing_mode();
        let entries = commands
            .into_entries()
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .collect::<Vec<_>>();
        if u32::try_from(entries.len()).is_err() {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract("CUDA command count exceeds u32"),
            ));
        }
        let command_phases = entries
            .iter()
            .map(|(phase, _, _)| *phase)
            .collect::<Vec<_>>();
        let command_node_indices = timing_mode.kernel_attribution_enabled().then(|| {
            entries
                .iter()
                .map(|(_, node_index, _)| *node_index)
                .collect::<Vec<_>>()
        });
        let commands = entries
            .into_iter()
            .map(|(_, _, command)| command)
            .collect::<Vec<_>>();
        let kernel_attribution = timing_mode.kernel_attribution_enabled();
        let mut execution_paths =
            kernel_attribution.then(|| vec![DeviceExecutionPath::Eager; commands.len()]);
        if commands
            .iter()
            .any(|command| command.runtime_instance != self.runtime_instance)
        {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract(
                    "CUDA command batch contains work from another runtime instance",
                ),
            ));
        }
        if let (Some(command_node_indices), Some(execution_paths)) =
            (&command_node_indices, &execution_paths)
        {
            if let Err(error) = cuda_submission_attribution(
                &command_phases,
                command_node_indices,
                &commands,
                execution_paths,
            ) {
                return Err(DefinitelyNotSubmitted::new(error));
            }
        }
        if let Err(error) = self.context.bind_to_thread() {
            return Err(DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::driver(
                "submission context binding",
                error,
            )));
        }
        let executable_candidates = match cuda_executable_candidates(&command_phases, &commands) {
            Ok(candidates) => candidates,
            Err(error) => return Err(DefinitelyNotSubmitted::new(error)),
        };
        let capture_allowed = stream.state.is_quiescent();
        if let Err(error) = stream.state.begin_submission() {
            return Err(DefinitelyNotSubmitted::new(error));
        }
        let mut replay_observation = DeviceReusableExecutionObservation::default();
        if S::ENABLED {
            for _ in &executable_candidates {
                replay_observation.observe_candidate_segment();
            }
        }
        let preparation = stream.executable_cache.prepare_all(
            &self.context,
            &stream.stream,
            &stream.blas,
            &commands,
            &executable_candidates,
            capture_allowed,
        );
        match preparation {
            Ok(preparation) if S::ENABLED => {
                for _ in 0..preparation.captured_segments() {
                    replay_observation.observe_captured_segment();
                }
                for _ in 0..preparation.uploaded_segments() {
                    replay_observation.observe_uploaded_segment();
                }
                for _ in 0..preparation.cache_hit_segments() {
                    replay_observation.observe_cache_hit_segment();
                }
                for _ in 0..preparation.cached_rejected_segments() {
                    replay_observation.observe_cached_rejected_segment();
                }
                for _ in 0..preparation.capture_rejected_segments() {
                    replay_observation.observe_capture_rejection();
                }
                for _ in 0..preparation.quiescence_deferred_segments() {
                    replay_observation.observe_quiescence_deferred_segment();
                }
                for _ in 0..preparation.capacity_deferred_segments() {
                    replay_observation.observe_capacity_deferred_segment();
                }
                for _ in 0..preparation.outside_preparation_segments() {
                    replay_observation.observe_outside_preparation_segment();
                }
                for _ in 0..preparation.evicted_segments() {
                    replay_observation.observe_evicted_segment();
                }
            }
            Ok(_) => {}
            Err(error) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while preparing reusable executables: {error}"
                );
            }
        }
        drop(validate_stage);

        let begin_timing_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::BeginTiming);
        let timing = match timing_mode {
            DeviceTimingMode::Off => CudaFenceTiming::NotRequested,
            DeviceTimingMode::Completion | DeviceTimingMode::Kernel => {
                match stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                {
                    Ok(start) => CudaFenceTiming::Events { start },
                    Err(_) => CudaFenceTiming::Unavailable,
                }
            }
        };
        drop(begin_timing_stage);

        let enqueue_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::EnqueueCommands);
        let mut command_events = kernel_attribution.then(|| Vec::with_capacity(commands.len()));
        let mut command_timing_unavailable = None;
        let mut index = 0;
        let mut executable_candidate_index = 0;
        while index < commands.len() {
            while executable_candidates
                .get(executable_candidate_index)
                .is_some_and(|candidate| candidate.start() < index)
            {
                executable_candidate_index += 1;
            }
            let replayed_end = executable_candidates
                .get(executable_candidate_index)
                .filter(|candidate| candidate.start() == index)
                .and_then(|candidate| {
                    match stream.executable_cache.launch(&stream.stream, candidate) {
                        Ok(true) => Some(Ok(candidate.end())),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                });
            match replayed_end {
                Some(Ok(segment_end)) => {
                    if let Some(execution_paths) = execution_paths.as_mut() {
                        execution_paths[index..segment_end].fill(DeviceExecutionPath::Replayed);
                    }
                    if kernel_attribution {
                        command_events = None;
                        command_timing_unavailable =
                            Some(DeviceTimingUnavailableReason::BackendUnsupported);
                    }
                    if S::ENABLED {
                        replay_observation.observe_replayed_segment(segment_end - index);
                    }
                    index = segment_end;
                    continue;
                }
                Some(Err(error)) => {
                    stream.state.fail();
                    self.quarantine(stream, commands);
                    panic!(
                        "CUDA submission became indeterminate while launching a reusable executable: {error}"
                    );
                }
                None => {}
            }
            let command_start = command_events.as_ref().and_then(|_| {
                stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .ok()
            });
            if let Err(error) = commands[index].enqueue(&stream.stream, &stream.blas) {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!("CUDA submission became indeterminate while enqueueing its batch: {error}");
            }
            if command_events.is_some() {
                let command_end = stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .ok();
                match command_start.zip(command_end) {
                    Some((start, end)) => {
                        let command = &commands[index];
                        let kind = if command.compute_dispatch_count > 0 {
                            DeviceExecutionIntervalKind::Compute
                        } else {
                            DeviceExecutionIntervalKind::Transfer
                        };
                        command_events
                            .as_mut()
                            .expect("CUDA command event collection was checked")
                            .push(CudaCommandEventTiming {
                                command_index: index as u32,
                                kind,
                                operation: command.operation,
                                start,
                                end,
                            });
                    }
                    None => {
                        command_events = None;
                        command_timing_unavailable =
                            Some(DeviceTimingUnavailableReason::BackendMeasurementFailed);
                    }
                }
            }
            if S::ENABLED {
                replay_observation.observe_eager_command();
            }
            index += 1;
        }
        drop(enqueue_stage);
        if S::ENABLED {
            timing_sink.record_reusable_execution(replay_observation);
        }
        let attribution = match command_node_indices
            .as_ref()
            .zip(execution_paths.as_ref())
            .map(|(command_node_indices, execution_paths)| {
                cuda_submission_attribution(
                    &command_phases,
                    command_node_indices,
                    &commands,
                    execution_paths,
                )
            }) {
            None => None,
            Some(Ok(attribution)) => Some(attribution),
            Some(Err(error)) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while binding native attribution: {error}"
                );
            }
        };
        let command_timing = match timing_mode {
            DeviceTimingMode::Off | DeviceTimingMode::Completion => {
                CudaFenceCommandTiming::NotRequested
            }
            DeviceTimingMode::Kernel => match command_timing_unavailable {
                Some(reason) => CudaFenceCommandTiming::Unavailable(reason),
                None => command_events.map_or(
                    CudaFenceCommandTiming::Unavailable(
                        DeviceTimingUnavailableReason::BackendMeasurementFailed,
                    ),
                    CudaFenceCommandTiming::Events,
                ),
            },
        };

        let fence_stage = CudaSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::RecordFenceAndAccount,
        );
        let fence_flags = match timing_mode {
            DeviceTimingMode::Off => None,
            DeviceTimingMode::Completion | DeviceTimingMode::Kernel => {
                Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)
            }
        };
        let event = match stream.stream.record_event(fence_flags) {
            Ok(event) => event,
            Err(error) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!("CUDA submission became indeterminate while recording its fence: {error:?}");
            }
        };
        if let Err(error) = stream.state.submission_recorded() {
            stream.state.fail();
            self.quarantine(stream, commands);
            panic!("CUDA submission became indeterminate while accounting its fence: {error}");
        }
        let fence = CudaDeviceFence {
            event,
            timing,
            command_timing,
            attribution,
            stream_state: Arc::clone(&stream.state),
            terminal_accounted: AtomicBool::new(false),
            _stream: Arc::clone(&stream.stream),
            _blas: Arc::clone(&stream.blas),
            _commands: commands,
        };
        drop(fence_stage);
        Ok(fence)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        fence.attribution.clone()
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        if let Err(error) = fence.event.context().bind_to_thread() {
            fence.stream_state.fail();
            return FenceQuery::Indeterminate(CudaDeviceRuntimeError::driver(
                "fence context binding",
                error,
            ));
        }
        match unsafe { cudarc::driver::result::event::query(fence.event.cu_event()) } {
            Ok(()) => {
                fence.mark_terminal();
                FenceQuery::Terminal(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                FenceQuery::Pending
            }
            Err(error) => {
                fence.stream_state.fail();
                FenceQuery::Indeterminate(CudaDeviceRuntimeError::driver("fence query", error))
            }
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        match fence.event.synchronize() {
            Ok(()) => {
                fence.mark_terminal();
                Ok(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            Err(error) => {
                fence.stream_state.fail();
                Err(FenceIndeterminate::new(CudaDeviceRuntimeError::driver(
                    "fence wait",
                    error,
                )))
            }
        }
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        self.validate_stream(stream)?;
        match stream.stream.synchronize() {
            Ok(()) => {
                self.release_quarantine(stream.id);
                stream.state.synchronized();
                Ok(())
            }
            Err(error) => {
                stream.state.fail();
                Err(CudaDeviceRuntimeError::driver(
                    "stream synchronization",
                    error,
                ))
            }
        }
    }

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        self.validate_stream(stream)?;
        self.validate_buffer(source)?;
        if output_layout.element_type() != source.descriptor.element_type {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA readback layout differs from source element type",
            ));
        }
        let output_bytes = output_layout
            .byte_len()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        let source_end = checked_end(
            region.source_offset_bytes(),
            region.length_bytes(),
            source.descriptor.size_bytes,
            "CUDA readback source",
        )?;
        let output_end = checked_end(
            region.destination_offset_bytes(),
            region.length_bytes(),
            output_bytes,
            "CUDA readback output",
        )?;
        self.synchronize(stream)?;
        let source_region = source.region(region.source_offset_bytes()..source_end)?;
        let mut output = vec![0_u8; checked_usize(output_bytes, "CUDA readback output")?];
        let output_start = checked_usize(
            region.destination_offset_bytes(),
            "CUDA readback output offset",
        )?;
        let output_end = checked_usize(output_end, "CUDA readback output end")?;
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                &mut output[output_start..output_end],
                source_region.device_ptr,
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("host readback", error))?;
        Ok(output)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        let (code, retryable) = match error {
            CudaDeviceRuntimeError::Blas { source, .. }
                if source.0 == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED =>
            {
                ("cuda_blas_allocation_failed", true)
            }
            CudaDeviceRuntimeError::Blas { .. } => ("cuda_blas_error", false),
            _ => match error.driver_code() {
                Some(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY) => {
                    ("cuda_out_of_memory", true)
                }
                Some(code) => (
                    match code {
                        cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => "cuda_not_ready",
                        cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_CONTEXT => {
                            "cuda_invalid_context"
                        }
                        cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS => {
                            "cuda_illegal_address"
                        }
                        _ => "cuda_driver_error",
                    },
                    false,
                ),
                None => ("cuda_runtime_contract", false),
            },
        };
        DeviceErrorReport::new(code, error.to_string(), retryable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::DeviceCommandPhase;

    fn command(operation: &'static str) -> CudaDeviceCommand {
        CudaDeviceCommand {
            runtime_instance: 1,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 1,
            token_count: 1,
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            executable: Arc::new(CudaCommandExecutable {
                regions: Vec::new(),
                host_storage: Vec::new(),
                enqueue: Mutex::new(Box::new(|_, _, _, _| Ok(()))),
            }),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
        }
    }

    #[test]
    fn kernel_attribution_retains_core_identity_and_cuda_work_shape() {
        let compute = command("test_compute")
            .with_work_attribution(DeviceBatchingForm::Packed, 2, 8, 3, 0)
            .unwrap();
        let binding = command("test_binding")
            .with_work_attribution(DeviceBatchingForm::ParticipantLoop, 2, 8, 0, 2)
            .unwrap();
        let attribution = cuda_submission_attribution(
            &[
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::DynamicBinding,
            ],
            &[Some(0), Some(0)],
            &[compute, binding],
            &[DeviceExecutionPath::Eager, DeviceExecutionPath::Replayed],
        )
        .unwrap();

        let [compute, binding] = attribution.commands() else {
            panic!("expected two CUDA attribution rows")
        };
        assert_eq!(compute.command_index(), 0);
        assert_eq!(compute.node_index(), Some(0));
        assert_eq!(compute.command_phase(), DeviceCommandPhase::Compute);
        assert_eq!(compute.native_op_id(), "test_compute");
        assert_eq!(compute.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(compute.batching_form(), DeviceBatchingForm::Packed);
        assert_eq!(compute.participant_count(), 2);
        assert_eq!(compute.token_count(), 8);
        assert_eq!(compute.compute_dispatch_count(), 3);
        assert_eq!(compute.transfer_command_count(), 0);

        assert_eq!(binding.command_index(), 1);
        assert_eq!(binding.command_phase(), DeviceCommandPhase::DynamicBinding);
        assert_eq!(binding.execution_path(), DeviceExecutionPath::Replayed);
        assert_eq!(binding.compute_dispatch_count(), 0);
        assert_eq!(binding.transfer_command_count(), 2);
    }

    #[test]
    fn cuda_work_attribution_rejects_empty_native_work() {
        let error = command("test_invalid")
            .with_work_attribution(DeviceBatchingForm::Scalar, 1, 1, 0, 0)
            .unwrap_err();
        assert!(error.to_string().contains("no participants or native work"));
    }
}
