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
#[cfg(feature = "vllm-marlin")]
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
#[cfg(feature = "vllm-marlin")]
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    BufferDescriptor, CapabilityId, CopyRegion, DefinitelyNotSubmitted, DeviceBatchingForm,
    DeviceBufferRetention, DeviceClass, DeviceCommandBatch, DeviceCommandEntry,
    DeviceCommandLogicalWork, DeviceCommandPhase, DeviceComputePathRequirement, DeviceDescriptor,
    DeviceErrorReport, DeviceExecutionInterval, DeviceExecutionIntervalKind, DeviceExecutionPath,
    DeviceExecutionSpanKind, DeviceExecutionTiming, DeviceId, DeviceNativeOperationId,
    DeviceNativeWorkAttribution, DeviceReplayedLogicalCommandAttribution,
    DeviceReplayedSegmentAttribution, DeviceReusableAddressScope, DeviceReusableExecutionCapture,
    DeviceReusableExecutionInvocation, DeviceReusableExecutionObservation,
    DeviceReusableExecutionPlan, DeviceReusableExecutionPreparation,
    DeviceReusableExecutionProgram, DeviceReusableExecutionProgramGapReason,
    DeviceReusableExecutionTrim, DeviceRuntime, DeviceSubmissionAttribution,
    DeviceSubmissionExecutionSpan, DeviceSubmissionExecutionTiming, DeviceSubmissionStage,
    DeviceSubmissionTimingSink, DeviceTerminal, DeviceTerminalReceipt, DeviceTimingMeasurement,
    DeviceTimingMode, DeviceTimingUnavailableReason, DisabledDeviceSubmissionTimingSink,
    DynamicStorageProfile, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout,
    ProgramBindingNodeBinding, RetainedHostMemoryRegion, StaticWeightTransformPlan,
    StaticWeightTransformRequest, StreamState, VNextError, DEVICE_COPY_NATIVE_OPERATION_ID,
    DEVICE_ZERO_NATIVE_OPERATION_ID, HOST_UPLOAD_NATIVE_OPERATION_ID,
};
use ferrum_types::AttentionExecutionPolicy;

use super::vnext_replay::{cuda_executable_candidates, CudaCommandReplayKey, CudaExecutableCache};
use super::vnext_tool_correlation;

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
    pub attention_execution_policy: AttentionExecutionPolicy,
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

pub(crate) struct CudaProgramBindingWrite {
    destination_offset_bytes: u64,
    payload: Box<[u8]>,
}

impl CudaProgramBindingWrite {
    pub(crate) fn new(
        destination_offset_bytes: u64,
        payload: Box<[u8]>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if payload.is_empty() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA program binding write payload is empty",
            ));
        }
        Ok(Self {
            destination_offset_bytes,
            payload,
        })
    }
}

struct CudaProgramBindingPatch {
    binding: ProgramBindingNodeBinding,
    destination: CudaBufferRegion,
    writes: Vec<CudaProgramBindingWrite>,
    fence_dependencies: Vec<CudaBufferRegion>,
}

struct CudaProgramBindingTransfer {
    destination_offset_bytes: u64,
    destination_stride_bytes: u64,
    row_bytes: usize,
    row_count: usize,
    payload: Box<[u8]>,
}

fn coalesce_program_binding_transfers(
    mut writes: Vec<CudaProgramBindingWrite>,
    arena_size_bytes: u64,
) -> Result<Vec<CudaProgramBindingTransfer>, CudaDeviceRuntimeError> {
    if writes.is_empty() || arena_size_bytes == 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA sparse program binding transfer has no writes or arena",
        ));
    }
    writes.sort_by_key(|write| write.destination_offset_bytes);

    let mut prior_end = 0_u64;
    for write in &writes {
        let payload_bytes = u64::try_from(write.payload.len()).map_err(|_| {
            CudaDeviceRuntimeError::contract("CUDA program binding payload exceeds u64")
        })?;
        if payload_bytes == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA sparse program binding write payload is empty",
            ));
        }
        let end = write
            .destination_offset_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA sparse program binding write range overflows u64",
                )
            })?;
        if write.destination_offset_bytes < prior_end || end > arena_size_bytes {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA sparse program binding writes overlap or exceed the arena",
            ));
        }
        prior_end = end;
    }

    let mut groups = Vec::new();
    let mut group_count = 0_usize;
    let mut group_bytes = 0_usize;
    let mut group_end = None;
    for write in &writes {
        if group_end.is_some_and(|end| end != write.destination_offset_bytes) {
            groups.push((group_count, group_bytes));
            group_count = 0;
            group_bytes = 0;
        }
        group_count = group_count.checked_add(1).ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA sparse program binding transfer count overflows usize",
            )
        })?;
        group_bytes = group_bytes
            .checked_add(write.payload.len())
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA sparse program binding transfer size overflows usize",
                )
            })?;
        group_end = Some(
            write
                .destination_offset_bytes
                .checked_add(u64::try_from(write.payload.len()).map_err(|_| {
                    CudaDeviceRuntimeError::contract("CUDA program binding payload exceeds u64")
                })?)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding write range overflows u64",
                    )
                })?,
        );
    }
    groups.push((group_count, group_bytes));

    let mut writes = writes.into_iter();
    let mut rows = Vec::with_capacity(groups.len());
    for (group_count, group_bytes) in groups {
        let first = writes
            .next()
            .expect("validated sparse transfer group owns its first write");
        let destination_offset_bytes = first.destination_offset_bytes;
        if group_count == 1 {
            rows.push(CudaProgramBindingWrite {
                destination_offset_bytes,
                payload: first.payload,
            });
            continue;
        }
        let mut payload = Vec::with_capacity(group_bytes);
        payload.extend_from_slice(&first.payload);
        for _ in 1..group_count {
            let write = writes
                .next()
                .expect("validated sparse transfer group owns every adjacent write");
            payload.extend_from_slice(&write.payload);
        }
        debug_assert_eq!(payload.len(), group_bytes);
        rows.push(CudaProgramBindingWrite {
            destination_offset_bytes,
            payload: payload.into_boxed_slice(),
        });
    }
    debug_assert!(writes.next().is_none());

    let mut transfers = Vec::with_capacity(rows.len());
    let mut row_index = 0_usize;
    while row_index < rows.len() {
        let row_bytes = rows[row_index].payload.len();
        let mut row_count = 1_usize;
        let mut destination_stride_bytes = u64::try_from(row_bytes).map_err(|_| {
            CudaDeviceRuntimeError::contract("CUDA sparse program binding row size exceeds u64")
        })?;
        if let Some(next) = rows.get(row_index + 1).filter(|next| {
            next.payload.len() == row_bytes
                && next.destination_offset_bytes > rows[row_index].destination_offset_bytes
        }) {
            destination_stride_bytes = next
                .destination_offset_bytes
                .checked_sub(rows[row_index].destination_offset_bytes)
                .expect("sorted non-overlapping program binding rows have a positive stride");
            row_count = 2;
            while let Some(next) = rows.get(row_index + row_count) {
                let prior = &rows[row_index + row_count - 1];
                if next.payload.len() != row_bytes
                    || next
                        .destination_offset_bytes
                        .checked_sub(prior.destination_offset_bytes)
                        != Some(destination_stride_bytes)
                {
                    break;
                }
                row_count += 1;
            }
        }

        let packed_bytes = row_bytes.checked_mul(row_count).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA sparse program binding packed rows exceed usize")
        })?;
        let destination_offset_bytes = rows[row_index].destination_offset_bytes;
        let payload = if row_count == 1 {
            std::mem::take(&mut rows[row_index].payload)
        } else {
            let mut payload = Vec::with_capacity(packed_bytes);
            for row in &rows[row_index..row_index + row_count] {
                payload.extend_from_slice(&row.payload);
            }
            debug_assert_eq!(payload.len(), packed_bytes);
            payload.into_boxed_slice()
        };
        transfers.push(CudaProgramBindingTransfer {
            destination_offset_bytes,
            destination_stride_bytes,
            row_bytes,
            row_count,
            payload,
        });
        row_index += row_count;
    }
    Ok(transfers)
}

/// Encoded CUDA work. Buffer and host-transfer storage stays alive until the
/// returned fence reaches a terminal state.
pub struct CudaDeviceCommand {
    runtime_instance: u64,
    operation: &'static str,
    batching_form: DeviceBatchingForm,
    participant_start: u32,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    executable: Option<Arc<CudaCommandExecutable>>,
    fence_dependencies: Vec<CudaBufferRegion>,
    replay_key: Option<CudaCommandReplayKey>,
    reusable_address_scope: Option<DeviceReusableAddressScope>,
    replay_gap_reason: Option<DeviceReusableExecutionProgramGapReason>,
    program_binding_patch: Option<CudaProgramBindingPatch>,
    reusable_execution: Option<DeviceReusableExecutionInvocation>,
}

impl fmt::Debug for CudaDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceCommand")
            .field("runtime_instance", &self.runtime_instance)
            .field("operation", &self.operation)
            .field("batching_form", &self.batching_form)
            .field("participant_start", &self.participant_start)
            .field("participant_count", &self.participant_count)
            .field("token_count", &self.token_count)
            .field("compute_dispatch_count", &self.compute_dispatch_count)
            .field("transfer_command_count", &self.transfer_command_count)
            .field(
                "captured_region_count",
                &self
                    .executable
                    .as_ref()
                    .map_or(0, |executable| executable.regions.len()),
            )
            .field(
                "captured_host_storage_count",
                &self
                    .executable
                    .as_ref()
                    .map_or(0, |executable| executable.host_storage.len()),
            )
            .field("fence_dependency_count", &self.fence_dependencies.len())
            .field("replayable", &self.replay_key.is_some())
            .field(
                "typed_program_binding_patch",
                &self.program_binding_patch.is_some(),
            )
            .field(
                "direct_reusable_execution",
                &self.reusable_execution.is_some(),
            )
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

    /// Encodes eager work while retaining additional submission-scoped
    /// allocations through the completion fence. Fence dependencies are not
    /// executable inputs and never make the command replayable.
    pub(crate) fn operation_with_blas_and_fence_dependencies(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        fence_dependencies: Vec<CudaBufferRegion>,
        enqueue: impl Fn(&CudaStream, &CudaBlas, &[CudaBufferRegion]) -> Result<(), CudaDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::operation_inner(
            operation,
            regions,
            fence_dependencies,
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

    pub(crate) fn replayable_operation_with_host_storage_blas_and_fence_dependencies(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        fence_dependencies: Vec<CudaBufferRegion>,
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
            fence_dependencies,
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
        let (replay_key, reusable_address_scope, replay_gap_reason) =
            bind_replay_contract(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: Some(Arc::new(CudaCommandExecutable {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            })),
            fence_dependencies,
            replay_key,
            reusable_address_scope,
            replay_gap_reason,
            program_binding_patch: None,
            reusable_execution: None,
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
        let (replay_key, reusable_address_scope, replay_gap_reason) =
            bind_replay_contract(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: Some(Arc::new(CudaCommandExecutable {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            })),
            fence_dependencies,
            replay_key,
            reusable_address_scope,
            replay_gap_reason,
            program_binding_patch: None,
            reusable_execution: None,
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
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            executable: Some(executable),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: None,
        }
    }

    pub(crate) fn program_binding_patch(
        operation: &'static str,
        binding: ProgramBindingNodeBinding,
        destination: CudaBufferRegion,
        mut writes: Vec<CudaProgramBindingWrite>,
        fence_dependencies: Vec<CudaBufferRegion>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let runtime_instance = destination.runtime_instance;
        validate_fence_dependencies(runtime_instance, &fence_dependencies)?;
        let slot = binding.slot();
        if destination.element_type != ElementType::U8
            || destination.length_bytes == 0
            || destination.length_bytes > slot.capacity_size_bytes()
            || writes.is_empty()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA program binding destination differs from its compiled slot",
            ));
        }
        writes.sort_by_key(|write| write.destination_offset_bytes);
        let mut prior_end = 0_u64;
        for write in &writes {
            let payload_bytes = u64::try_from(write.payload.len()).map_err(|_| {
                CudaDeviceRuntimeError::contract("CUDA program binding payload exceeds u64")
            })?;
            let end = write
                .destination_offset_bytes
                .checked_add(payload_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA program binding write range overflows u64",
                    )
                })?;
            if write.destination_offset_bytes < prior_end || end > destination.length_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA program binding writes overlap or exceed the logical slot",
                ));
            }
            prior_end = end;
        }
        Ok(Self {
            runtime_instance,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            executable: None,
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: Some(CudaProgramBindingPatch {
                binding,
                destination,
                writes,
                fence_dependencies,
            }),
            reusable_execution: None,
        })
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
        self.participant_start = 0;
        self.participant_count = participant_count;
        self.token_count = token_count;
        self.compute_dispatch_count = compute_dispatch_count;
        self.transfer_command_count = transfer_command_count;
        Ok(self)
    }

    fn with_initialization_native_work(
        mut self,
        compute_dispatch_count: u64,
        transfer_command_count: u64,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if compute_dispatch_count == 0 || transfer_command_count == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA static transform attribution requires compute and transfer work",
            ));
        }
        self.compute_dispatch_count = compute_dispatch_count;
        self.transfer_command_count = transfer_command_count;
        Ok(self)
    }

    fn bind_core_logical_work(
        mut self,
        logical_work: DeviceCommandLogicalWork,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if self.participant_count != 0 || self.token_count != 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA core logical work cannot replace provider command attribution",
            ));
        }
        self.batching_form = logical_work.batching_form();
        self.participant_start = logical_work.participant_start();
        self.participant_count = logical_work.participant_count();
        self.token_count = logical_work.token_count();
        Ok(self)
    }

    fn reusable_execution(
        runtime_instance: u64,
        invocation: DeviceReusableExecutionInvocation,
    ) -> Self {
        let participant_count = invocation.participant_count();
        let token_count = invocation.token_count();
        let executable = Arc::new(CudaCommandExecutable {
            regions: Vec::new(),
            host_storage: Vec::new(),
            enqueue: Mutex::new(Box::new(|_, _, _, _| {
                Err(CudaDeviceRuntimeError::contract(
                    "direct reusable execution must be resolved by the owning stream cache",
                ))
            })),
        });
        Self {
            runtime_instance,
            operation: "vnext_reusable_execution",
            batching_form: DeviceBatchingForm::ParticipantLoop,
            participant_start: 0,
            participant_count,
            token_count,
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            executable: Some(executable),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: Some(invocation),
        }
    }

    fn coalesced_program_bindings(
        mut commands: Vec<Self>,
    ) -> Result<Vec<Self>, CudaDeviceRuntimeError> {
        if commands.is_empty() {
            return Ok(commands);
        }
        let typed_patch_count = commands
            .iter()
            .filter(|command| command.program_binding_patch.is_some())
            .count();
        if typed_patch_count == 0 {
            if commands.len() == 1 {
                return Ok(commands);
            }
            return Self::coalesced_opaque_program_bindings(commands);
        }
        if typed_patch_count != commands.len() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA program binding prelude mixes typed and opaque patches",
            ));
        }

        let runtime_instance = commands[0].runtime_instance;
        let participant_start = commands[0].participant_start;
        let participant_count = commands[0].participant_count;
        let token_count = commands[0].token_count;
        if participant_count == 0
            || commands.iter().any(|command| {
                command.runtime_instance != runtime_instance
                    || command.participant_start != participant_start
                    || command.participant_count != participant_count
                    || command.token_count != token_count
                    || command.compute_dispatch_count != 0
                    || command.transfer_command_count == 0
                    || command.replay_key.is_some()
                    || command.reusable_address_scope.is_some()
            })
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA typed program bindings are not one compatible prelude",
            ));
        }

        let mut patches = commands
            .iter_mut()
            .map(|command| {
                command.program_binding_patch.take().ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA typed program binding patch disappeared during coalescing",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        patches.sort_by_key(|patch| patch.binding.node_index());
        let first = patches.first().expect("non-empty typed patch set");
        let layout = first.binding.layout();
        let lane_slot = first.binding.lane_slot_identity();
        let plan_hash = first.binding.plan_hash();
        if patches.len() != layout.slots().len()
            || patches.iter().zip(layout.slots()).any(|(patch, slot)| {
                patch.binding.node_index() != slot.node_index()
                    || patch.binding.plan_hash() != plan_hash
                    || patch.binding.layout().fingerprint() != layout.fingerprint()
                    || patch.binding.lane_slot_identity() != lane_slot
                    || patch.binding.slot() != slot
            })
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA typed program bindings do not cover one compiled layout exactly",
            ));
        }

        let layout_physical_size_bytes = layout.physical_size_bytes();
        let first_destination = first.destination.clone();
        let first_slot_offset_bytes = first.binding.slot().physical_offset_bytes();
        let arena_device_ptr = first_destination
            .device_ptr
            .checked_sub(first_slot_offset_bytes)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA program binding arena base pointer underflows",
                )
            })?;
        let allocation_end = first_destination
            ._allocation
            .aligned_ptr
            .checked_add(first_destination._allocation.requested_bytes)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA program binding allocation end overflows")
            })?;
        let arena_end = arena_device_ptr
            .checked_add(layout_physical_size_bytes)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA program binding arena end overflows")
            })?;
        if arena_device_ptr < first_destination._allocation.aligned_ptr
            || arena_end > allocation_end
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA compiled program binding arena exceeds its admitted allocation",
            ));
        }

        let mut fence_dependencies = Vec::new();
        let mut arena_writes = Vec::new();
        for patch in patches {
            let slot = patch.binding.slot();
            let expected_device_ptr = arena_device_ptr
                .checked_add(slot.physical_offset_bytes())
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("CUDA program binding slot pointer overflows")
                })?;
            if patch.destination.runtime_instance != runtime_instance
                || patch.destination.device_ptr != expected_device_ptr
                || patch.destination.element_type != ElementType::U8
                || !Arc::ptr_eq(
                    &patch.destination._allocation,
                    &first_destination._allocation,
                )
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA program binding patch destination differs from its arena slot",
                ));
            }
            for mut write in patch.writes {
                write.destination_offset_bytes = slot
                    .physical_offset_bytes()
                    .checked_add(write.destination_offset_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA sparse program binding write offset overflows",
                        )
                    })?;
                arena_writes.push(write);
            }
            fence_dependencies.extend(patch.fence_dependencies);
        }

        let transfers =
            coalesce_program_binding_transfers(arena_writes, layout_physical_size_bytes)?;
        let transfer_command_count = u64::try_from(transfers.len()).map_err(|_| {
            CudaDeviceRuntimeError::contract(
                "CUDA sparse program binding transfer count exceeds u64",
            )
        })?;
        let mut regions = Vec::with_capacity(transfers.len());
        let mut host_storage = Vec::with_capacity(transfers.len());
        let mut transfer_shapes = Vec::with_capacity(transfers.len());
        for transfer in transfers {
            let row_bytes_u64 = u64::try_from(transfer.row_bytes).map_err(|_| {
                CudaDeviceRuntimeError::contract("CUDA sparse program binding row size exceeds u64")
            })?;
            let trailing_rows =
                u64::try_from(transfer.row_count.saturating_sub(1)).map_err(|_| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding row count exceeds u64",
                    )
                })?;
            let destination_span_bytes = transfer
                .destination_stride_bytes
                .checked_mul(trailing_rows)
                .and_then(|span| span.checked_add(row_bytes_u64))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding destination span overflows",
                    )
                })?;
            let destination_end = transfer
                .destination_offset_bytes
                .checked_add(destination_span_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding destination end overflows",
                    )
                })?;
            if destination_end > layout_physical_size_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA sparse program binding transfer exceeds its arena",
                ));
            }
            let device_ptr = arena_device_ptr
                .checked_add(transfer.destination_offset_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding destination pointer overflows",
                    )
                })?;
            regions.push(CudaBufferRegion {
                _allocation: Arc::clone(&first_destination._allocation),
                _core_retention: first_destination._core_retention.clone(),
                reusable_address_scope: first_destination.reusable_address_scope,
                runtime_instance,
                device_ptr,
                length_bytes: destination_span_bytes,
                element_type: ElementType::U8,
            });
            transfer_shapes.push((
                checked_usize(
                    transfer.destination_stride_bytes,
                    "CUDA sparse program binding destination stride",
                )?,
                transfer.row_bytes,
                transfer.row_count,
            ));
            host_storage.push(transfer.payload);
        }
        let executable = Arc::new(CudaCommandExecutable {
            regions,
            host_storage,
            enqueue: Mutex::new(Box::new(move |stream, _blas, regions, host_storage| {
                if regions.len() != host_storage.len() || regions.len() != transfer_shapes.len() {
                    return Err(CudaDeviceRuntimeError::contract(
                        "CUDA sparse program binding transfer storage differs from its shape",
                    ));
                }
                for ((region, payload), &(destination_pitch, row_bytes, row_count)) in
                    regions.iter().zip(host_storage).zip(&transfer_shapes)
                {
                    if row_count == 1 {
                        unsafe {
                            cudarc::driver::result::memcpy_htod_async(
                                region.device_ptr,
                                payload.as_ref(),
                                stream.cu_stream(),
                            )
                        }
                        .map_err(|error| {
                            CudaDeviceRuntimeError::driver("sparse program binding upload", error)
                        })?;
                        continue;
                    }
                    let copy = cudarc::driver::sys::CUDA_MEMCPY2D {
                        srcXInBytes: 0,
                        srcY: 0,
                        srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: payload.as_ptr().cast(),
                        srcDevice: 0,
                        srcArray: std::ptr::null_mut(),
                        srcPitch: row_bytes,
                        dstXInBytes: 0,
                        dstY: 0,
                        dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstHost: std::ptr::null_mut(),
                        dstDevice: region.device_ptr,
                        dstArray: std::ptr::null_mut(),
                        dstPitch: destination_pitch,
                        WidthInBytes: row_bytes,
                        Height: row_count,
                    };
                    unsafe { cudarc::driver::sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) }
                        .result()
                        .map_err(|error| {
                            CudaDeviceRuntimeError::driver(
                                "strided sparse program binding upload",
                                error,
                            )
                        })?;
                }
                Ok(())
            })),
        });
        Ok(vec![Self {
            runtime_instance,
            operation: "vnext_program_binding_prelude",
            batching_form: DeviceBatchingForm::ParticipantLoop,
            participant_start,
            participant_count,
            token_count,
            compute_dispatch_count: 0,
            transfer_command_count,
            executable: Some(executable),
            fence_dependencies,
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: None,
        }])
    }

    fn coalesced_opaque_program_bindings(
        commands: Vec<Self>,
    ) -> Result<Vec<Self>, CudaDeviceRuntimeError> {
        let runtime_instance = commands[0].runtime_instance;
        let participant_start = commands[0].participant_start;
        let participant_count = commands[0].participant_count;
        let token_count = commands[0].token_count;
        if participant_count == 0
            || commands.iter().any(|command| {
                command.runtime_instance != runtime_instance
                    || command.participant_start != participant_start
                    || command.participant_count != participant_count
                    || command.token_count != token_count
                    || command.compute_dispatch_count != 0
                    || command.transfer_command_count == 0
                    || command.replay_key.is_some()
                    || command.reusable_address_scope.is_some()
            })
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA program bindings are not one compatible eager prelude",
            ));
        }
        let transfer_command_count = commands.iter().try_fold(0_u64, |total, command| {
            total
                .checked_add(command.transfer_command_count)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA program binding transfer count overflows u64",
                    )
                })
        })?;
        let executable = Arc::new(CudaCommandExecutable {
            regions: Vec::new(),
            host_storage: Vec::new(),
            enqueue: Mutex::new(Box::new(move |stream, blas, _regions, _host_storage| {
                commands
                    .iter()
                    .try_for_each(|command| command.enqueue(stream, blas))
            })),
        });
        Ok(vec![Self {
            runtime_instance,
            operation: "vnext_program_binding_prelude",
            batching_form: DeviceBatchingForm::ParticipantLoop,
            participant_start,
            participant_count,
            token_count,
            compute_dispatch_count: 0,
            transfer_command_count,
            executable: Some(executable),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: None,
        }])
    }

    pub(crate) fn enqueue(
        &self,
        stream: &CudaStream,
        blas: &CudaBlas,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let executable = self.executable.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "uncoalesced CUDA program binding patch cannot enqueue",
            )
        })?;
        let enqueue = executable
            .enqueue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enqueue(stream, blas, &executable.regions, &executable.host_storage)
    }

    pub(crate) const fn replay_key(&self) -> Option<CudaCommandReplayKey> {
        self.replay_key
    }

    pub(crate) fn reusable_execution_invocation(
        &self,
    ) -> Option<&DeviceReusableExecutionInvocation> {
        self.reusable_execution.as_ref()
    }

    pub(crate) fn replayed_logical_attribution(
        &self,
        logical_command_ordinal: u32,
        node_index: u32,
        reusable_graph_node_count: u32,
    ) -> Option<DeviceReplayedLogicalCommandAttribution> {
        DeviceReplayedLogicalCommandAttribution::new(
            logical_command_ordinal,
            node_index,
            DeviceNativeOperationId::new(self.operation)?,
            self.batching_form,
            self.participant_count,
            self.token_count,
            self.compute_dispatch_count,
            self.transfer_command_count,
            u64::from(reusable_graph_node_count),
        )
    }

    pub(crate) const fn reusable_address_scope(&self) -> Option<DeviceReusableAddressScope> {
        self.reusable_address_scope
    }

    pub(crate) const fn replay_gap_reason(
        &self,
    ) -> Option<DeviceReusableExecutionProgramGapReason> {
        self.replay_gap_reason
    }

    pub(crate) fn executable(&self) -> Arc<CudaCommandExecutable> {
        Arc::clone(
            self.executable
                .as_ref()
                .expect("replayable CUDA command owns an executable"),
        )
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
    Option<DeviceReusableExecutionProgramGapReason>,
) {
    let Some(key) = replay_key else {
        return (
            None,
            None,
            Some(DeviceReusableExecutionProgramGapReason::ProviderReplayKeyMissing),
        );
    };
    let mut scope = DeviceReusableAddressScope::Plan;
    for region in regions {
        let Some(region_scope) = region.reusable_address_scope else {
            return (
                None,
                None,
                Some(DeviceReusableExecutionProgramGapReason::ReusableAddressScopeMissing),
            );
        };
        match region_scope {
            DeviceReusableAddressScope::Plan => {}
            DeviceReusableAddressScope::ExecutionLane(lane_id) => {
                match scope {
                    DeviceReusableAddressScope::Plan => {
                        scope = DeviceReusableAddressScope::ExecutionLane(lane_id);
                    }
                    DeviceReusableAddressScope::ExecutionLane(current) if current == lane_id => {}
                    DeviceReusableAddressScope::ExecutionLane(_) => return (
                        None,
                        None,
                        Some(DeviceReusableExecutionProgramGapReason::ReusableAddressScopeConflict),
                    ),
                }
            }
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
        None,
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
    reusable_graph_node_counts: Option<&[Option<u64>]>,
    replayed_segments: Vec<DeviceReplayedSegmentAttribution>,
) -> Result<DeviceSubmissionAttribution, CudaDeviceRuntimeError> {
    if command_phases.len() != commands.len()
        || command_node_indices.len() != commands.len()
        || execution_paths.len() != commands.len()
        || reusable_graph_node_counts.is_some_and(|counts| counts.len() != commands.len())
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
            let native_op_id =
                DeviceNativeOperationId::new(command.operation).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA command attribution has a non-portable native operation identity",
                    )
                })?;
            DeviceNativeWorkAttribution::with_participant_range(
                command_index,
                command_node_indices[command_index as usize],
                command_phases[command_index as usize],
                native_op_id,
                execution_paths[command_index as usize],
                command.batching_form,
                command.participant_start,
                command.participant_count,
                command.token_count,
                command.compute_dispatch_count,
                command.transfer_command_count,
                reusable_graph_node_counts.and_then(|counts| counts[command_index as usize]),
            )
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA command attribution has invalid native work metadata",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    DeviceSubmissionAttribution::with_replayed_segments(rows, replayed_segments).ok_or_else(|| {
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

impl CudaFenceTiming {
    fn origin(&self) -> Option<&CudaEvent> {
        match self {
            Self::Events { start } => Some(start),
            Self::NotRequested | Self::Unavailable => None,
        }
    }
}

enum CudaExecutionSpanEventMeasurement {
    Events { start: CudaEvent, end: CudaEvent },
    Unavailable(DeviceTimingUnavailableReason),
}

struct CudaExecutionSpanEventTiming {
    start_command_index: u32,
    end_command_index: u32,
    span_kind: DeviceExecutionSpanKind,
    interval_kind: DeviceExecutionIntervalKind,
    operation: &'static str,
    reusable_executable_fingerprint: Option<Arc<str>>,
    measurement: CudaExecutionSpanEventMeasurement,
}

impl CudaExecutionSpanEventTiming {
    fn new(
        start_command_index: usize,
        end_command_index: usize,
        span_kind: DeviceExecutionSpanKind,
        interval_kind: DeviceExecutionIntervalKind,
        operation: &'static str,
        reusable_executable_fingerprint: Option<Arc<str>>,
        events: Option<(CudaEvent, CudaEvent)>,
    ) -> Option<Self> {
        let start_command_index = u32::try_from(start_command_index).ok()?;
        let end_command_index = u32::try_from(end_command_index).ok()?;
        let measurement = events.map_or(
            CudaExecutionSpanEventMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            ),
            |(start, end)| CudaExecutionSpanEventMeasurement::Events { start, end },
        );
        Some(Self {
            start_command_index,
            end_command_index,
            span_kind,
            interval_kind,
            operation,
            reusable_executable_fingerprint,
            measurement,
        })
    }

    fn resolve(&self, origin: &CudaEvent) -> Option<DeviceSubmissionExecutionSpan> {
        let span = match &self.measurement {
            CudaExecutionSpanEventMeasurement::Events { start, end } => {
                let interval = cuda_event_elapsed_ns(origin, start)
                    .zip(cuda_event_elapsed_ns(origin, end))
                    .and_then(|(start_offset_ns, end_offset_ns)| {
                        DeviceExecutionInterval::new_labeled(
                            self.interval_kind,
                            start_offset_ns,
                            end_offset_ns,
                            self.operation,
                        )
                    });
                match interval {
                    Some(interval) => DeviceSubmissionExecutionSpan::measured(
                        self.start_command_index,
                        self.end_command_index,
                        self.span_kind,
                        vec![interval],
                    ),
                    None => DeviceSubmissionExecutionSpan::unavailable(
                        self.start_command_index,
                        self.end_command_index,
                        self.span_kind,
                        DeviceTimingUnavailableReason::BackendMeasurementFailed,
                    ),
                }
            }
            CudaExecutionSpanEventMeasurement::Unavailable(reason) => {
                DeviceSubmissionExecutionSpan::unavailable(
                    self.start_command_index,
                    self.end_command_index,
                    self.span_kind,
                    *reason,
                )
            }
        }?;
        match &self.reusable_executable_fingerprint {
            Some(fingerprint) => {
                span.with_reusable_executable_fingerprint(fingerprint.as_ref().to_owned())
            }
            None => Some(span),
        }
    }
}

enum CudaFenceCommandTiming {
    NotRequested,
    Unavailable(DeviceTimingUnavailableReason),
    Spans {
        command_count: u32,
        spans: Vec<CudaExecutionSpanEventTiming>,
    },
}

impl CudaFenceCommandTiming {
    fn measurement(
        &self,
        origin: Option<&CudaEvent>,
    ) -> DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        match self {
            Self::NotRequested => DeviceTimingMeasurement::NotRequested,
            Self::Unavailable(reason) => DeviceTimingMeasurement::Unavailable(*reason),
            Self::Spans {
                command_count,
                spans,
            } => {
                let Some(origin) = origin else {
                    return DeviceTimingMeasurement::Unavailable(
                        DeviceTimingUnavailableReason::BackendMeasurementFailed,
                    );
                };
                let spans = spans
                    .iter()
                    .map(|span| span.resolve(origin))
                    .collect::<Option<Vec<_>>>()
                    .and_then(|spans| {
                        DeviceSubmissionExecutionTiming::from_spans(*command_count, spans)
                    });
                spans.map_or_else(
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
                    CudaFenceCommandTiming::Unavailable(_)
                    | CudaFenceCommandTiming::Spans { .. } => {
                        DeviceTerminalReceipt::profiled_with_submission_timing(
                            terminal,
                            self.execution_timing(),
                            self.command_timing.measurement(self.timing.origin()),
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

#[cfg(feature = "vllm-marlin")]
const MXFP4_BLOCKS_TO_GPTQ_WORDS_FUNCTION: &str = "gpt_oss_mxfp4_blocks_to_gptq_words";
#[cfg(feature = "vllm-marlin")]
const MXFP4_SCALES_TO_MARLIN_FUNCTION: &str = "gpt_oss_mxfp4_scales_to_marlin";

#[cfg(feature = "vllm-marlin")]
#[derive(Clone)]
struct Mxfp4MarlinPrepareFunctions {
    blocks_to_gptq_words: CudaFunction,
    scales_to_marlin: CudaFunction,
}

#[cfg(feature = "vllm-marlin")]
impl Mxfp4MarlinPrepareFunctions {
    fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        let module = context
            .load_module(Ptx::from_src(crate::ptx::MXFP4_MARLIN_PREPARE.to_owned()))
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 prepare module load", error)
            })?;
        let blocks_to_gptq_words = module
            .load_function(MXFP4_BLOCKS_TO_GPTQ_WORDS_FUNCTION)
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 block transpose load", error)
            })?;
        let scales_to_marlin = module
            .load_function(MXFP4_SCALES_TO_MARLIN_FUNCTION)
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 scale prepare load", error)
            })?;
        Ok(Self {
            blocks_to_gptq_words,
            scales_to_marlin,
        })
    }
}

/// Concrete CUDA primitive runtime consumed by the shared vNext resource and
/// operation dispatch layers.
pub struct CudaDeviceRuntime {
    descriptor: DeviceDescriptor,
    attention_execution_policy: AttentionExecutionPolicy,
    runtime_instance: u64,
    context: Arc<CudaContext>,
    allocation_stream: Arc<CudaStream>,
    #[cfg(feature = "vllm-marlin")]
    mxfp4_marlin_prepare: Mxfp4MarlinPrepareFunctions,
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
        if !config.attention_execution_policy.is_resolved() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime requires a resolved attention execution policy",
            ));
        }
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
        #[cfg(feature = "vllm-marlin")]
        let mxfp4_marlin_prepare = Mxfp4MarlinPrepareFunctions::load(&context)?;
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
            attention_execution_policy: config.attention_execution_policy,
            runtime_instance,
            context,
            allocation_stream,
            #[cfg(feature = "vllm-marlin")]
            mxfp4_marlin_prepare,
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

#[cfg(feature = "vllm-marlin")]
const BLOCK_FP8_GROUP128_STATIC_TRANSFORM_OPERATION: &str =
    "static_weight.block_fp8_to_marlin_fp8_group128";
#[cfg(feature = "vllm-marlin")]
const GPT_OSS_MXFP4_STATIC_TRANSFORM_OPERATION: &str = "static_weight.gpt_oss_mxfp4_to_marlin";

#[cfg(feature = "vllm-marlin")]
#[derive(Clone, Copy)]
struct BlockFp8Group128CudaTransform {
    size_n: u64,
    size_k: u64,
    matrices_per_output: usize,
    source_matrix_count: usize,
    output_matrix_count: usize,
    value_matrix_bytes: u64,
    source_scale_matrix_bytes: u64,
    packed_matrix_bytes: u64,
    scale_matrix_bytes: u64,
}

#[cfg(feature = "vllm-marlin")]
#[derive(Clone, Copy)]
struct GptOssMxfp4CudaTransform {
    expert_count: u64,
    size_n: i32,
    size_k: i32,
    expert_packed_bytes: u64,
    expert_scale_bytes: u64,
    expert_packed_host_bytes: usize,
    expert_scale_host_bytes: usize,
    packed_grid: u32,
    scale_grid: u32,
}

#[cfg(feature = "vllm-marlin")]
fn checked_product(
    values: impl IntoIterator<Item = u64>,
    context: &'static str,
) -> Result<u64, CudaDeviceRuntimeError> {
    values.into_iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| CudaDeviceRuntimeError::contract(format!("{context} overflows u64")))
    })
}

#[cfg(feature = "vllm-marlin")]
fn validate_static_transform_regions(
    packed: &CudaBufferRegion,
    scales: &CudaBufferRegion,
    scratch: &CudaBufferRegion,
) -> Result<(), CudaDeviceRuntimeError> {
    if packed.device_ptr() % 4 != 0 || scales.device_ptr() % 2 != 0 || scratch.device_ptr() % 4 != 0
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA static weight transform has a misaligned packed, scale, or scratch address",
        ));
    }
    let regions = [packed, scales, scratch];
    for left in 0..regions.len() {
        let left_end = regions[left]
            .device_ptr()
            .checked_add(regions[left].length_bytes())
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA static weight transform region address overflows",
                )
            })?;
        for right in left + 1..regions.len() {
            let right_end = regions[right]
                .device_ptr()
                .checked_add(regions[right].length_bytes())
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA static weight transform region address overflows",
                    )
                })?;
            if regions[left].device_ptr() < right_end && regions[right].device_ptr() < left_end {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA static weight transform packed, scale, and scratch regions must not overlap",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "vllm-marlin")]
fn encode_block_fp8_group128_static_transform(
    runtime: &CudaDeviceRuntime,
    request: StaticWeightTransformRequest<'_, '_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
    let (
        source_values_id,
        source_scales_id,
        packed_values_id,
        scales_id,
        logical_dimensions,
        matrices_per_output,
    ) = match request.plan() {
        StaticWeightTransformPlan::BlockFp8ToMarlinFp8Group128 {
            source_values_id,
            source_scales_id,
            packed_values_id,
            scales_id,
            logical_dimensions,
            matrices_per_output,
        } => (
            source_values_id,
            source_scales_id,
            packed_values_id,
            scales_id,
            logical_dimensions,
            *matrices_per_output,
        ),
        StaticWeightTransformPlan::GptOssMxfp4ToMarlin { .. } => {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA block-FP8 encoder received a GPT-OSS MXFP4 transform plan",
            ));
        }
    };

    if logical_dimensions.len() < 2 || !matches!(matrices_per_output, 1 | 2) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform has an unsupported rank or fusion count",
        ));
    }
    let rank = logical_dimensions.len();
    let size_n = logical_dimensions[rank - 2];
    let size_k = logical_dimensions[rank - 1];
    if size_n == 0
        || size_k == 0
        || !size_n.is_multiple_of(128)
        || !size_k.is_multiple_of(128)
        || (matrices_per_output == 2 && (rank < 4 || logical_dimensions[rank - 3] != 2))
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform requires positive 128-aligned matrices and typed gate/up fusion",
        ));
    }

    let source_matrix_count_u64 = checked_product(
        logical_dimensions[..rank - 2].iter().copied(),
        "CUDA block-FP8 source matrix count",
    )?;
    if !source_matrix_count_u64.is_multiple_of(u64::from(matrices_per_output)) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 source matrix count is not divisible by its fusion count",
        ));
    }
    let source_matrix_count = checked_usize(
        source_matrix_count_u64,
        "CUDA block-FP8 source matrix count",
    )?;
    let matrices_per_output = usize::try_from(matrices_per_output).expect("1 or 2 fits usize");
    let output_matrix_count = source_matrix_count / matrices_per_output;
    if output_matrix_count == 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform has no output matrix",
        ));
    }

    let value_matrix_bytes = size_n.checked_mul(size_k).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA block-FP8 matrix byte size overflows u64")
    })?;
    let source_scale_matrix_bytes = (size_n / 128)
        .checked_mul(size_k / 128)
        .and_then(|elements| elements.checked_mul(ElementType::Bf16.size_bytes()))
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA block-FP8 source scale matrix byte size overflows u64",
            )
        })?;
    let fused_n = size_n
        .checked_mul(matrices_per_output as u64)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA block-FP8 fused N overflows u64"))?;
    let packed_matrix_bytes = fused_n.checked_mul(size_k).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA block-FP8 packed matrix byte size overflows u64")
    })?;
    let scale_matrix_bytes = fused_n
        .checked_mul(size_k / 128)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA block-FP8 destination scale matrix byte size overflows u64",
            )
        })?;
    let packed_total_bytes = packed_matrix_bytes
        .checked_mul(output_matrix_count as u64)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA block-FP8 packed stack size overflows u64")
        })?;
    let scales_total_bytes = scale_matrix_bytes
        .checked_mul(output_matrix_count as u64)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA block-FP8 scale stack size overflows u64")
        })?;

    let sources = request.sources();
    let destinations = request.destinations();
    if sources.len() != 2
        || destinations.len() != 2
        || sources[0].component_id() != source_values_id
        || sources[1].component_id() != source_scales_id
        || destinations[0].component().id != *packed_values_id
        || destinations[1].component().id != *scales_id
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform source or destination identity/order differs from its plan",
        ));
    }

    let mut expected_source_scale_dimensions = logical_dimensions.clone();
    expected_source_scale_dimensions[rank - 2] /= 128;
    expected_source_scale_dimensions[rank - 1] /= 128;
    let mut expected_destination_scale_dimensions = logical_dimensions.clone();
    expected_destination_scale_dimensions[rank - 1] /= 128;
    if sources[0].dimensions() != logical_dimensions.as_slice()
        || sources[1].dimensions() != expected_source_scale_dimensions.as_slice()
        || destinations[0].component().dimensions != *logical_dimensions
        || destinations[1].component().dimensions != expected_destination_scale_dimensions
        || sources[0].element_type() != ElementType::U8
        || sources[1].element_type() != ElementType::Bf16
        || destinations[0].component().physical_element_type() != ElementType::U8
        || destinations[1].component().physical_element_type() != ElementType::F16
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform source/destination shape or element type differs from the group-128 ABI",
        ));
    }
    let source_values_total_bytes = value_matrix_bytes
        .checked_mul(source_matrix_count_u64)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA block-FP8 source stack size overflows u64")
        })?;
    let source_scales_total_bytes = source_scale_matrix_bytes
        .checked_mul(source_matrix_count_u64)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA block-FP8 source scale stack size overflows u64")
        })?;
    if sources[0].total_bytes() != source_values_total_bytes
        || sources[1].total_bytes() != source_scales_total_bytes
        || destinations[0]
            .component()
            .physical_bytes()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?
            != packed_total_bytes
        || destinations[1]
            .component()
            .physical_bytes()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?
            != scales_total_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform component byte extents differ from the exact product ABI",
        ));
    }

    let value_segment_bytes =
        checked_usize(value_matrix_bytes, "CUDA block-FP8 value segment byte size")?;
    let scale_segment_bytes = checked_usize(
        source_scale_matrix_bytes,
        "CUDA block-FP8 scale segment byte size",
    )?;
    if sources[0].segments().len() != source_matrix_count
        || sources[1].segments().len() != source_matrix_count
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform requires one ordered retained source segment per matrix",
        ));
    }
    let mut retained_sources = Vec::with_capacity(source_matrix_count.saturating_mul(2));
    for (matrix, segment) in sources[0].segments().iter().enumerate() {
        let retained = segment.retained_host_memory().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(format!(
                "CUDA block-FP8 value matrix {matrix} lacks retained stable host memory"
            ))
        })?;
        if segment.bytes().len() != value_segment_bytes
            || retained.length_bytes() != value_segment_bytes
            || !std::ptr::eq(segment.bytes().as_ptr(), retained.bytes().as_ptr())
        {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA block-FP8 value matrix {matrix} segment differs from its exact retained range"
            )));
        }
        retained_sources.push(retained.clone());
    }
    for (matrix, segment) in sources[1].segments().iter().enumerate() {
        let retained = segment.retained_host_memory().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(format!(
                "CUDA block-FP8 scale matrix {matrix} lacks retained stable host memory"
            ))
        })?;
        if segment.bytes().len() != scale_segment_bytes
            || retained.length_bytes() != scale_segment_bytes
            || !std::ptr::eq(segment.bytes().as_ptr(), retained.bytes().as_ptr())
        {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA block-FP8 scale matrix {matrix} segment differs from its exact retained range"
            )));
        }
        for (block, bytes) in segment.bytes().chunks_exact(2).enumerate() {
            let inverse_scale = half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
            let marlin_scale = half::f16::from_f32(inverse_scale * 256.0);
            if !inverse_scale.is_finite()
                || inverse_scale <= 0.0
                || !marlin_scale.is_finite()
                || marlin_scale == half::f16::ZERO
            {
                return Err(CudaDeviceRuntimeError::contract(format!(
                    "CUDA block-FP8 scale matrix {matrix} block {block} cannot be represented by the group-128 Marlin F16 ABI"
                )));
            }
        }
        retained_sources.push(retained.clone());
    }

    let packed_destination = &destinations[0];
    let scales_destination = &destinations[1];
    let scratch = request.scratch();
    runtime.validate_buffer(packed_destination.buffer())?;
    runtime.validate_buffer(scales_destination.buffer())?;
    runtime.validate_buffer(scratch)?;
    if packed_destination.buffer().descriptor.element_type != ElementType::U8
        || scales_destination.buffer().descriptor.element_type != ElementType::F16
        || scratch.descriptor.element_type != ElementType::U8
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform buffer element types differ from U8/F16/U8",
        ));
    }
    let admitted_scratch_bytes = request
        .plan()
        .scratch_bytes()
        .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
    if admitted_scratch_bytes != packed_matrix_bytes
        || scratch.descriptor.size_bytes < admitted_scratch_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform scratch differs from its single fused-matrix plan",
        ));
    }
    let packed_end = checked_end(
        packed_destination.destination_offset_bytes(),
        packed_total_bytes,
        packed_destination.buffer().descriptor.size_bytes,
        "CUDA block-FP8 packed destination",
    )?;
    let scales_end = checked_end(
        scales_destination.destination_offset_bytes(),
        scales_total_bytes,
        scales_destination.buffer().descriptor.size_bytes,
        "CUDA block-FP8 scales destination",
    )?;
    let packed_region = packed_destination
        .buffer()
        .region(packed_destination.destination_offset_bytes()..packed_end)?;
    let scales_region = scales_destination
        .buffer()
        .region(scales_destination.destination_offset_bytes()..scales_end)?;
    let scratch_region = scratch.region(0..admitted_scratch_bytes)?;
    validate_static_transform_regions(&packed_region, &scales_region, &scratch_region)?;

    let transform = BlockFp8Group128CudaTransform {
        size_n,
        size_k,
        matrices_per_output,
        source_matrix_count,
        output_matrix_count,
        value_matrix_bytes,
        source_scale_matrix_bytes,
        packed_matrix_bytes,
        scale_matrix_bytes,
    };
    let compute_dispatch_count = (output_matrix_count as u64).checked_mul(2).ok_or_else(|| {
        CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform dispatch count overflows u64",
        )
    })?;
    let transfer_command_count = source_matrix_count_u64.checked_mul(2).ok_or_else(|| {
        CudaDeviceRuntimeError::contract(
            "CUDA block-FP8 static transform transfer count overflows u64",
        )
    })?;
    let command = CudaDeviceCommand::operation(
        BLOCK_FP8_GROUP128_STATIC_TRANSFORM_OPERATION,
        vec![packed_region, scales_region, scratch_region],
        move |stream, regions| {
            debug_assert_eq!(regions.len(), 3);
            for output_matrix in 0..transform.output_matrix_count {
                for matrix_lane in 0..transform.matrices_per_output {
                    let source_matrix = output_matrix * transform.matrices_per_output + matrix_lane;
                    let scratch_offset = (matrix_lane as u64)
                        .checked_mul(transform.value_matrix_bytes)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "CUDA block-FP8 scratch value offset overflows",
                            )
                        })?;
                    let scratch_pointer = regions[2]
                        .device_ptr()
                        .checked_add(scratch_offset)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "CUDA block-FP8 scratch value pointer overflows",
                            )
                        })?;
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(
                            scratch_pointer,
                            retained_sources[source_matrix].bytes(),
                            stream.cu_stream(),
                        )
                    }
                    .map_err(|error| {
                        CudaDeviceRuntimeError::driver("block-FP8 value upload", error)
                    })?;
                }
                let packed_offset = (output_matrix as u64)
                    .checked_mul(transform.packed_matrix_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA block-FP8 packed destination offset overflows",
                        )
                    })?;
                let packed_pointer = regions[0]
                    .device_ptr()
                    .checked_add(packed_offset)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA block-FP8 packed destination pointer overflows",
                        )
                    })?;
                unsafe {
                    super::vllm_marlin::launch_block_fp8_group128_repack(
                        stream,
                        regions[2].device_ptr(),
                        packed_pointer,
                        transform.size_k,
                        transform.size_n * transform.matrices_per_output as u64,
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;

                for matrix_lane in 0..transform.matrices_per_output {
                    let source_matrix = output_matrix * transform.matrices_per_output + matrix_lane;
                    let scratch_offset = (matrix_lane as u64)
                        .checked_mul(transform.source_scale_matrix_bytes)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "CUDA block-FP8 scratch scale offset overflows",
                            )
                        })?;
                    let scratch_pointer = regions[2]
                        .device_ptr()
                        .checked_add(scratch_offset)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "CUDA block-FP8 scratch scale pointer overflows",
                            )
                        })?;
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(
                            scratch_pointer,
                            retained_sources[transform.source_matrix_count + source_matrix].bytes(),
                            stream.cu_stream(),
                        )
                    }
                    .map_err(|error| {
                        CudaDeviceRuntimeError::driver("block-FP8 scale upload", error)
                    })?;
                }
                let scales_offset = (output_matrix as u64)
                    .checked_mul(transform.scale_matrix_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA block-FP8 scale destination offset overflows",
                        )
                    })?;
                let scales_pointer = regions[1]
                    .device_ptr()
                    .checked_add(scales_offset)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA block-FP8 scale destination pointer overflows",
                        )
                    })?;
                unsafe {
                    super::vllm_marlin::launch_block_fp8_group128_scales(
                        stream,
                        regions[2].device_ptr(),
                        scales_pointer,
                        transform.size_k,
                        transform.size_n * transform.matrices_per_output as u64,
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
            }
            Ok(())
        },
    )?;
    command.with_initialization_native_work(compute_dispatch_count, transfer_command_count)
}

#[cfg(feature = "vllm-marlin")]
fn encode_gpt_oss_mxfp4_static_transform(
    runtime: &CudaDeviceRuntime,
    request: StaticWeightTransformRequest<'_, '_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
    let (source_blocks_id, source_scales_id, packed_values_id, scales_id, logical_dimensions) =
        match request.plan() {
            StaticWeightTransformPlan::GptOssMxfp4ToMarlin {
                source_blocks_id,
                source_scales_id,
                packed_values_id,
                scales_id,
                logical_dimensions,
            } => (
                source_blocks_id,
                source_scales_id,
                packed_values_id,
                scales_id,
                logical_dimensions,
            ),
            StaticWeightTransformPlan::BlockFp8ToMarlinFp8Group128 { .. } => {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA GPT-OSS MXFP4 encoder received a block-FP8 transform plan",
                ));
            }
        };
    let [expert_count, size_n_u64, size_k_u64] = logical_dimensions.as_slice() else {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform requires exact [E,N,K] dimensions",
        ));
    };
    let expert_count = *expert_count;
    let size_n_u64 = *size_n_u64;
    let size_k_u64 = *size_k_u64;
    if expert_count == 0
        || size_n_u64 == 0
        || size_k_u64 == 0
        || !size_n_u64.is_multiple_of(64)
        || !size_k_u64.is_multiple_of(64)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform requires positive E and 64-aligned N/K",
        ));
    }
    let size_n = i32::try_from(size_n_u64)
        .map_err(|_| CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 N exceeds native i32"))?;
    let size_k = i32::try_from(size_k_u64)
        .map_err(|_| CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 K exceeds native i32"))?;
    let expert_packed_bytes = size_n_u64
        .checked_mul(size_k_u64)
        .and_then(|elements| elements.checked_div(2))
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA GPT-OSS MXFP4 expert packed byte size overflows u64",
            )
        })?;
    let expert_scale_bytes = size_n_u64.checked_mul(size_k_u64 / 32).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 expert scale byte size overflows u64")
    })?;
    let packed_total_bytes = expert_packed_bytes
        .checked_mul(expert_count)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 packed stack size overflows u64")
        })?;
    let scales_total_bytes = expert_scale_bytes
        .checked_mul(expert_count)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 scale stack size overflows u64")
        })?;

    let sources = request.sources();
    let destinations = request.destinations();
    if sources.len() != 2
        || destinations.len() != 2
        || sources[0].component_id() != source_blocks_id
        || sources[1].component_id() != source_scales_id
        || destinations[0].component().id != *packed_values_id
        || destinations[1].component().id != *scales_id
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform source or destination identity/order differs from its plan",
        ));
    }
    let expected_blocks_dimensions = [expert_count, size_n_u64, size_k_u64 / 32, 16];
    let expected_scales_dimensions = [expert_count, size_n_u64, size_k_u64 / 32];
    let expected_packed_dimensions = [expert_count, size_n_u64, size_k_u64 / 2];
    if sources[0].dimensions() != expected_blocks_dimensions
        || sources[1].dimensions() != expected_scales_dimensions
        || destinations[0].component().dimensions != expected_packed_dimensions
        || destinations[1].component().dimensions != expected_scales_dimensions
        || sources[0].element_type() != ElementType::U8
        || sources[1].element_type() != ElementType::U8
        || destinations[0].component().physical_element_type() != ElementType::U8
        || destinations[1].component().physical_element_type() != ElementType::U8
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform shape or element type differs from the exact source/Marlin ABI",
        ));
    }
    if sources[0].total_bytes() != packed_total_bytes
        || sources[1].total_bytes() != scales_total_bytes
        || destinations[0]
            .component()
            .physical_bytes()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?
            != packed_total_bytes
        || destinations[1]
            .component()
            .physical_bytes()
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?
            != scales_total_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform component byte extents differ from the exact ABI",
        ));
    }
    if sources[0].segments().len() != 1 || sources[1].segments().len() != 1 {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform requires one retained mmap segment per source tensor",
        ));
    }
    let block_segment = &sources[0].segments()[0];
    let scale_segment = &sources[1].segments()[0];
    let retained_blocks = block_segment.retained_host_memory().ok_or_else(|| {
        CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 blocks lack retained stable host memory",
        )
    })?;
    let retained_scales = scale_segment.retained_host_memory().ok_or_else(|| {
        CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 scales lack retained stable host memory",
        )
    })?;
    let packed_total_host_bytes = checked_usize(
        packed_total_bytes,
        "CUDA GPT-OSS MXFP4 packed source byte size",
    )?;
    let scales_total_host_bytes = checked_usize(
        scales_total_bytes,
        "CUDA GPT-OSS MXFP4 scale source byte size",
    )?;
    if block_segment.bytes().len() != packed_total_host_bytes
        || retained_blocks.length_bytes() != packed_total_host_bytes
        || !std::ptr::eq(
            block_segment.bytes().as_ptr(),
            retained_blocks.bytes().as_ptr(),
        )
        || scale_segment.bytes().len() != scales_total_host_bytes
        || retained_scales.length_bytes() != scales_total_host_bytes
        || !std::ptr::eq(
            scale_segment.bytes().as_ptr(),
            retained_scales.bytes().as_ptr(),
        )
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 source segments differ from their retained mmap ranges",
        ));
    }

    let packed_destination = &destinations[0];
    let scales_destination = &destinations[1];
    let scratch = request.scratch();
    runtime.validate_buffer(packed_destination.buffer())?;
    runtime.validate_buffer(scales_destination.buffer())?;
    runtime.validate_buffer(scratch)?;
    if packed_destination.buffer().descriptor.element_type != ElementType::U8
        || scales_destination.buffer().descriptor.element_type != ElementType::U8
        || scratch.descriptor.element_type != ElementType::U8
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 transform buffers must all use U8 storage",
        ));
    }
    let admitted_scratch_bytes = request
        .plan()
        .scratch_bytes()
        .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
    if admitted_scratch_bytes != expert_packed_bytes
        || scratch.descriptor.size_bytes < admitted_scratch_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA GPT-OSS MXFP4 scratch differs from its one-expert bounded plan",
        ));
    }
    let packed_end = checked_end(
        packed_destination.destination_offset_bytes(),
        packed_total_bytes,
        packed_destination.buffer().descriptor.size_bytes,
        "CUDA GPT-OSS MXFP4 packed destination",
    )?;
    let scales_end = checked_end(
        scales_destination.destination_offset_bytes(),
        scales_total_bytes,
        scales_destination.buffer().descriptor.size_bytes,
        "CUDA GPT-OSS MXFP4 scales destination",
    )?;
    let packed_region = packed_destination
        .buffer()
        .region(packed_destination.destination_offset_bytes()..packed_end)?;
    let scales_region = scales_destination
        .buffer()
        .region(scales_destination.destination_offset_bytes()..scales_end)?;
    let scratch_region = scratch.region(0..admitted_scratch_bytes)?;
    validate_static_transform_regions(&packed_region, &scales_region, &scratch_region)?;

    let packed_word_count = expert_packed_bytes / 4;
    let packed_grid = u32::try_from(packed_word_count.div_ceil(256)).map_err(|_| {
        CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 packed grid exceeds u32")
    })?;
    let scale_grid = u32::try_from(expert_scale_bytes.div_ceil(256)).map_err(|_| {
        CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 scale grid exceeds u32")
    })?;
    let transform = GptOssMxfp4CudaTransform {
        expert_count,
        size_n,
        size_k,
        expert_packed_bytes,
        expert_scale_bytes,
        expert_packed_host_bytes: checked_usize(
            expert_packed_bytes,
            "CUDA GPT-OSS MXFP4 expert packed byte size",
        )?,
        expert_scale_host_bytes: checked_usize(
            expert_scale_bytes,
            "CUDA GPT-OSS MXFP4 expert scale byte size",
        )?,
        packed_grid,
        scale_grid,
    };
    let compute_dispatch_count = expert_count.checked_mul(3).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 dispatch count overflows u64")
    })?;
    let transfer_command_count = expert_count.checked_mul(2).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("CUDA GPT-OSS MXFP4 transfer count overflows u64")
    })?;
    let retained_blocks = retained_blocks.clone();
    let retained_scales = retained_scales.clone();
    let functions = runtime.mxfp4_marlin_prepare.clone();
    let command = CudaDeviceCommand::operation(
        GPT_OSS_MXFP4_STATIC_TRANSFORM_OPERATION,
        vec![packed_region, scales_region, scratch_region],
        move |stream, regions| {
            debug_assert_eq!(regions.len(), 3);
            for expert in 0..transform.expert_count {
                let packed_offset = expert
                    .checked_mul(transform.expert_packed_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 expert packed offset overflows",
                        )
                    })?;
                let scale_offset = expert
                    .checked_mul(transform.expert_scale_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 expert scale offset overflows",
                        )
                    })?;
                let packed_pointer = regions[0]
                    .device_ptr()
                    .checked_add(packed_offset)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 packed pointer overflows",
                        )
                    })?;
                let scale_pointer = regions[1]
                    .device_ptr()
                    .checked_add(scale_offset)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 scale pointer overflows",
                        )
                    })?;
                let expert_index = usize::try_from(expert).map_err(|_| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA GPT-OSS MXFP4 expert index exceeds host address space",
                    )
                })?;
                let block_start = expert_index
                    .checked_mul(transform.expert_packed_host_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 host block offset overflows",
                        )
                    })?;
                let block_end = block_start
                    .checked_add(transform.expert_packed_host_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 host block end overflows",
                        )
                    })?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        packed_pointer,
                        &retained_blocks.bytes()[block_start..block_end],
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 block upload", error)
                })?;
                let scratch_pointer = regions[2].device_ptr();
                let mut transpose = stream.launch_builder(&functions.blocks_to_gptq_words);
                transpose.arg(&packed_pointer);
                transpose.arg(&scratch_pointer);
                transpose.arg(&transform.size_n);
                transpose.arg(&transform.size_k);
                unsafe {
                    transpose.launch(LaunchConfig {
                        grid_dim: (transform.packed_grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 block transpose", error)
                })?;
                unsafe {
                    super::vllm_marlin::vllm_gptq_marlin_repack_raw(
                        stream,
                        scratch_pointer,
                        packed_pointer,
                        transform.size_k,
                        transform.size_n,
                    )
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::contract(format!(
                        "GPT-OSS MXFP4 native Marlin repack failed: {error}"
                    ))
                })?;

                let scale_start = expert_index
                    .checked_mul(transform.expert_scale_host_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 host scale offset overflows",
                        )
                    })?;
                let scale_end = scale_start
                    .checked_add(transform.expert_scale_host_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "CUDA GPT-OSS MXFP4 host scale end overflows",
                        )
                    })?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        scratch_pointer,
                        &retained_scales.bytes()[scale_start..scale_end],
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 scale upload", error)
                })?;
                let mut scales = stream.launch_builder(&functions.scales_to_marlin);
                scales.arg(&scratch_pointer);
                scales.arg(&scale_pointer);
                scales.arg(&transform.size_n);
                scales.arg(&transform.size_k);
                unsafe {
                    scales.launch(LaunchConfig {
                        grid_dim: (transform.scale_grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("GPT-OSS MXFP4 scale prepare", error)
                })?;
            }
            Ok(())
        },
    )?;
    command.with_initialization_native_work(compute_dispatch_count, transfer_command_count)
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

    fn attention_execution_policy(&self) -> AttentionExecutionPolicy {
        self.attention_execution_policy
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

    fn encode_static_weight_transform(
        &self,
        request: StaticWeightTransformRequest<'_, '_, Self::Buffer>,
    ) -> Option<Result<Self::Command, Self::Error>> {
        #[cfg(feature = "vllm-marlin")]
        {
            Some(match request.plan() {
                StaticWeightTransformPlan::BlockFp8ToMarlinFp8Group128 { .. } => {
                    encode_block_fp8_group128_static_transform(self, request)
                }
                StaticWeightTransformPlan::GptOssMxfp4ToMarlin { .. } => {
                    encode_gpt_oss_mxfp4_static_transform(self, request)
                }
            })
        }
        #[cfg(not(feature = "vllm-marlin"))]
        {
            let _ = request;
            None
        }
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

    fn reusable_execution_catalog(
        &self,
        stream: &Self::Stream,
    ) -> Result<Vec<DeviceReusableExecutionProgram>, Self::Error> {
        self.validate_stream(stream)?;
        if !stream.state.is_quiescent() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable execution catalog requires its quiescent owning stream",
            ));
        }
        stream
            .executable_cache
            .catalog()
            .map_err(CudaDeviceRuntimeError::contract)
    }

    fn encode_reusable_execution(
        &self,
        invocation: DeviceReusableExecutionInvocation,
    ) -> Result<Option<Self::Command>, Self::Error> {
        if invocation.program_id().runtime_implementation_fingerprint()
            != self.descriptor.runtime_implementation_fingerprint
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable execution reference targets another runtime implementation",
            ));
        }
        Ok(Some(CudaDeviceCommand::reusable_execution(
            self.runtime_instance,
            invocation,
        )))
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
            DEVICE_COPY_NATIVE_OPERATION_ID.as_str(),
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
            HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
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
            DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
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

    fn coalesce_program_bindings(
        &self,
        commands: Vec<Self::Command>,
    ) -> Result<Vec<Self::Command>, Self::Error> {
        CudaDeviceCommand::coalesced_program_bindings(commands)
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
        let compute_path_requirement = commands.compute_path_requirement();
        let declared_eager_compute_node_indices = commands
            .declared_eager_compute_node_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let declared_eager_compute_node_count =
            commands.declared_eager_compute_node_indices().len();
        let logical_attribution = commands
            .attribution_requirement()
            .logical_execution_path_required();
        let reusable_execution_capture = commands.reusable_execution_capture().cloned();
        let entries = commands
            .into_entries()
            .into_iter()
            .map(|entry| {
                let (phase, node_index, logical_work, command) = entry.into_parts();
                let command = match logical_work {
                    Some(logical_work) => command.bind_core_logical_work(logical_work)?,
                    None => command,
                };
                Ok((phase, node_index, command))
            })
            .collect::<Result<Vec<_>, CudaDeviceRuntimeError>>()
            .map_err(DefinitelyNotSubmitted::new)?;
        let declaration_shape_matches = match compute_path_requirement {
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                !declared_eager_compute_node_indices.is_empty()
                    && declared_eager_compute_node_indices.len()
                        == declared_eager_compute_node_count
            }
            _ => declared_eager_compute_node_indices.is_empty(),
        };
        let mut compute_command_count = 0_usize;
        let mut direct_compute_command_count = 0_usize;
        let mut observed_eager_compute_node_indices = BTreeSet::new();
        let mut exact_boundary_shape = true;
        for (phase, node_index, command) in &entries {
            if *phase != DeviceCommandPhase::Compute {
                continue;
            }
            compute_command_count += 1;
            if let Some(invocation) = command.reusable_execution_invocation() {
                direct_compute_command_count += 1;
                if declared_eager_compute_node_indices
                    .iter()
                    .any(|node_index| invocation.segment().contains_node(*node_index))
                {
                    exact_boundary_shape = false;
                }
            } else if compute_path_requirement
                == DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
            {
                exact_boundary_shape &= node_index.is_some_and(|node_index| {
                    declared_eager_compute_node_indices.contains(&node_index)
                        && observed_eager_compute_node_indices.insert(node_index)
                });
            }
        }
        let command_count = u32::try_from(entries.len()).map_err(|_| {
            DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::contract(
                "CUDA command count exceeds u32",
            ))
        })?;
        let command_phases = entries
            .iter()
            .map(|(phase, _, _)| *phase)
            .collect::<Vec<_>>();
        let command_node_indices = (timing_mode.kernel_attribution_enabled()
            || logical_attribution
            || reusable_execution_capture.is_some())
        .then(|| {
            entries
                .iter()
                .map(|(_, node_index, _)| *node_index)
                .collect::<Vec<_>>()
        });
        let commands = entries
            .into_iter()
            .map(|(_, _, command)| command)
            .collect::<Vec<_>>();
        let physical_span_attribution = timing_mode.physical_span_attribution_enabled();
        let kernel_attribution = timing_mode.kernel_attribution_enabled();
        let native_attribution = kernel_attribution || logical_attribution;
        if kernel_attribution {
            vnext_tool_correlation::prepare();
        }
        let mut execution_paths =
            native_attribution.then(|| vec![DeviceExecutionPath::Eager; commands.len()]);
        let mut reusable_graph_node_counts = native_attribution.then(|| vec![None; commands.len()]);
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
        let contains_direct_execution = direct_compute_command_count != 0;
        let compute_path_matches = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => true,
            DeviceComputePathRequirement::EagerOnly => direct_compute_command_count == 0,
            DeviceComputePathRequirement::ReplayedOnly => {
                compute_command_count > 0 && direct_compute_command_count == compute_command_count
            }
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
                declaration_shape_matches
                    && exact_boundary_shape
                    && direct_compute_command_count > 0
                    && direct_compute_command_count < compute_command_count
                    && observed_eager_compute_node_indices == declared_eager_compute_node_indices
            }
        };
        if !compute_path_matches {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract(
                    "CUDA compute commands do not satisfy the required execution path",
                ),
            ));
        }
        if reusable_execution_capture.is_some() && contains_direct_execution {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract(
                    "CUDA reusable execution capture cannot contain direct program references",
                ),
            ));
        }
        if kernel_attribution && contains_direct_execution {
            return Err(DefinitelyNotSubmitted::new(
                CudaDeviceRuntimeError::contract(
                    "CUDA kernel attribution requires full logical command encoding",
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
                reusable_graph_node_counts.as_deref(),
                Vec::new(),
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
        for invocation in commands
            .iter()
            .filter_map(CudaDeviceCommand::reusable_execution_invocation)
        {
            let resident = if logical_attribution {
                stream
                    .executable_cache
                    .contains_attributable_program_segment(invocation)
            } else {
                stream.executable_cache.contains_program_segment(invocation)
            };
            match resident {
                Ok(true) => {}
                Ok(false) => {
                    return Err(DefinitelyNotSubmitted::new(
                        CudaDeviceRuntimeError::contract(if logical_attribution {
                            "CUDA direct reusable execution lacks sealed logical attribution"
                        } else {
                            "CUDA direct reusable execution is not resident in the sealed catalog"
                        }),
                    ))
                }
                Err(error) => {
                    return Err(DefinitelyNotSubmitted::new(
                        CudaDeviceRuntimeError::contract(error.to_string()),
                    ))
                }
            }
        }
        let executable_candidates = match compute_path_requirement {
            DeviceComputePathRequirement::Adaptive => {
                let eager_boundary_node_indices = reusable_execution_capture
                    .as_ref()
                    .map(DeviceReusableExecutionCapture::eager_boundary_node_indices)
                    .unwrap_or_default();
                match cuda_executable_candidates(
                    &command_phases,
                    &commands,
                    command_node_indices.as_deref(),
                    eager_boundary_node_indices,
                ) {
                    Ok(candidates) => candidates,
                    Err(error) => return Err(DefinitelyNotSubmitted::new(error)),
                }
            }
            DeviceComputePathRequirement::EagerOnly
            | DeviceComputePathRequirement::ReplayedOnly
            | DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => Vec::new(),
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
        let preparation = match stream.executable_cache.prepare_all(
            &self.context,
            &stream.stream,
            &stream.blas,
            &commands,
            &executable_candidates,
            capture_allowed,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while preparing reusable executables: {error}"
                );
            }
        };
        if S::ENABLED {
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
        if let Some(capture) = reusable_execution_capture.as_ref() {
            let command_node_indices = command_node_indices
                .as_deref()
                .expect("reusable execution capture retained node attribution");
            if let Err(error) = stream.executable_cache.register_program(
                capture,
                &executable_candidates,
                &command_phases,
                command_node_indices,
                &commands,
                &preparation,
            ) {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!(
                    "CUDA submission became indeterminate while registering a reusable program: {error}"
                );
            }
        }
        drop(validate_stage);

        let begin_timing_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::BeginTiming);
        let timing = match timing_mode {
            DeviceTimingMode::Off => CudaFenceTiming::NotRequested,
            DeviceTimingMode::Completion
            | DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => {
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
        let mut command_spans =
            physical_span_attribution.then(|| Vec::with_capacity(commands.len()));
        let mut replayed_segments = logical_attribution.then(Vec::new);
        let mut index = 0;
        let mut executable_candidate_index = 0;
        while index < commands.len() {
            if let Some(invocation) = commands[index].reusable_execution_invocation() {
                let start = command_spans.as_ref().and_then(|_| {
                    stream
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()
                });
                let launched = stream.executable_cache.launch_program_segment(
                    &stream.stream,
                    invocation,
                    timing_mode,
                    logical_attribution,
                );
                let end = command_spans.as_ref().and_then(|_| {
                    stream
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()
                });
                match launched {
                    Ok(Some(launch)) => {
                        if let Some(replayed_segments) = replayed_segments.as_mut() {
                            let physical_command_index = u32::try_from(index).expect(
                                "CUDA direct replay command index was validated before submission",
                            );
                            let reusable_executable_fingerprint = launch
                                .reusable_executable_fingerprint()
                                .expect("attributable CUDA replay retained its fingerprint");
                            let logical_commands = launch
                                .replayed_logical_commands()
                                .expect("attributable CUDA replay retained its logical commands");
                            let reusable_graph_node_count =
                                logical_commands.iter().try_fold(0_u64, |total, command| {
                                    total.checked_add(command.reusable_graph_node_count())
                                });
                            let Some(reusable_graph_node_count) = reusable_graph_node_count else {
                                stream.state.fail();
                                self.quarantine(stream, commands);
                                panic!(
                                    "CUDA submission became indeterminate because replay graph attribution overflowed u64"
                                );
                            };
                            let replayed = DeviceReplayedSegmentAttribution::new(
                                physical_command_index,
                                invocation.program_id().clone(),
                                invocation.segment().clone(),
                                reusable_executable_fingerprint.to_string(),
                                logical_commands.as_ref().to_vec(),
                            );
                            let Some(replayed) = replayed else {
                                stream.state.fail();
                                self.quarantine(stream, commands);
                                panic!(
                                    "CUDA submission became indeterminate because sealed replay attribution drifted"
                                );
                            };
                            execution_paths
                                .as_mut()
                                .expect("logical CUDA attribution retained execution paths")
                                [index] = DeviceExecutionPath::Replayed;
                            reusable_graph_node_counts
                                .as_mut()
                                .expect("logical CUDA attribution retained graph counts")[index] =
                                Some(reusable_graph_node_count);
                            replayed_segments.push(replayed);
                        }
                        if let Some(command_spans) = command_spans.as_mut() {
                            command_spans.push(
                                CudaExecutionSpanEventTiming::new(
                                    index,
                                    index + 1,
                                    DeviceExecutionSpanKind::ReusableExecutable,
                                    DeviceExecutionIntervalKind::Compute,
                                    "cuda direct reusable executable",
                                    launch.reusable_executable_fingerprint(),
                                    start.zip(end),
                                )
                                .expect("CUDA direct replay index was validated as u32"),
                            );
                        }
                        if S::ENABLED {
                            replay_observation.observe_replayed_segment(
                                invocation.segment().logical_command_count() as usize,
                            );
                        }
                        index += 1;
                        continue;
                    }
                    Ok(None) => {
                        stream.state.fail();
                        self.quarantine(stream, commands);
                        panic!("CUDA reusable execution disappeared after pre-submit validation");
                    }
                    Err(error) => {
                        stream.state.fail();
                        self.quarantine(stream, commands);
                        panic!(
                            "CUDA submission became indeterminate while launching a reusable program: {error}"
                        );
                    }
                }
            }
            while executable_candidates
                .get(executable_candidate_index)
                .is_some_and(|candidate| candidate.start() < index)
            {
                executable_candidate_index += 1;
            }
            let replay_candidate = executable_candidates
                .get(executable_candidate_index)
                .filter(|candidate| candidate.start() == index);
            let replayed = match replay_candidate {
                Some(candidate)
                    if physical_span_attribution && stream.executable_cache.contains(candidate) =>
                {
                    let start = command_spans.as_ref().and_then(|_| {
                        stream
                            .stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .ok()
                    });
                    let launched =
                        stream
                            .executable_cache
                            .launch(&stream.stream, candidate, timing_mode);
                    let end = command_spans.as_ref().and_then(|_| {
                        stream
                            .stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .ok()
                    });
                    match launched {
                        Ok(Some(launch)) => Some(Ok((
                            candidate.end(),
                            start.zip(end),
                            launch.reusable_executable_fingerprint(),
                            launch.reusable_graph_node_counts(),
                        ))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(candidate) if !physical_span_attribution => {
                    match stream
                        .executable_cache
                        .launch(&stream.stream, candidate, timing_mode)
                    {
                        Ok(Some(_)) => Some(Ok((candidate.end(), None, None, None))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(_) | None => None,
            };
            match replayed {
                Some(Ok((
                    segment_end,
                    events,
                    reusable_executable_fingerprint,
                    graph_node_counts,
                ))) => {
                    if let Some(execution_paths) = execution_paths.as_mut() {
                        execution_paths[index..segment_end].fill(DeviceExecutionPath::Replayed);
                    }
                    if let (Some(target), Some(observed)) =
                        (reusable_graph_node_counts.as_mut(), graph_node_counts)
                    {
                        debug_assert_eq!(observed.len(), segment_end - index);
                        for (target, observed) in target[index..segment_end]
                            .iter_mut()
                            .zip(observed.iter().copied())
                        {
                            *target = Some(u64::from(observed));
                        }
                    }
                    if let Some(command_spans) = command_spans.as_mut() {
                        command_spans.push(
                            CudaExecutionSpanEventTiming::new(
                                index,
                                segment_end,
                                DeviceExecutionSpanKind::ReusableExecutable,
                                DeviceExecutionIntervalKind::Compute,
                                "cuda reusable executable",
                                reusable_executable_fingerprint,
                                events,
                            )
                            .expect("CUDA replay range was validated as u32"),
                        );
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
            let command_start = command_spans.as_ref().and_then(|_| {
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
            if let Some(command_spans) = command_spans.as_mut() {
                let command_end = stream
                    .stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .ok();
                let command = &commands[index];
                let interval_kind = if command.compute_dispatch_count > 0 {
                    DeviceExecutionIntervalKind::Compute
                } else {
                    DeviceExecutionIntervalKind::Transfer
                };
                command_spans.push(
                    CudaExecutionSpanEventTiming::new(
                        index,
                        index + 1,
                        DeviceExecutionSpanKind::EagerCommand,
                        interval_kind,
                        command.operation,
                        None,
                        command_start.zip(command_end),
                    )
                    .expect("CUDA eager command index was validated as u32"),
                );
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
                    reusable_graph_node_counts.as_deref(),
                    replayed_segments.unwrap_or_default(),
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
            DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => command_spans.map_or(
                CudaFenceCommandTiming::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                ),
                |spans| CudaFenceCommandTiming::Spans {
                    command_count,
                    spans,
                },
            ),
        };

        let fence_stage = CudaSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::RecordFenceAndAccount,
        );
        let fence_flags = match timing_mode {
            DeviceTimingMode::Off => None,
            DeviceTimingMode::Completion
            | DeviceTimingMode::Replay
            | DeviceTimingMode::Kernel
            | DeviceTimingMode::Verification => {
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

    fn program_binding_write(offset: u64, payload: Vec<u8>) -> CudaProgramBindingWrite {
        CudaProgramBindingWrite::new(offset, payload.into_boxed_slice()).unwrap()
    }

    #[test]
    fn sparse_program_binding_transfers_preserve_live_bytes_and_destination_offsets() {
        let transfers = coalesce_program_binding_transfers(
            vec![
                program_binding_write(16, vec![9, 10]),
                program_binding_write(2, vec![3]),
                program_binding_write(0, vec![1, 2]),
            ],
            32,
        )
        .unwrap();

        assert_eq!(transfers.len(), 2);
        assert_eq!(transfers[0].destination_offset_bytes, 0);
        assert_eq!(transfers[0].destination_stride_bytes, 3);
        assert_eq!(transfers[0].row_bytes, 3);
        assert_eq!(transfers[0].row_count, 1);
        assert_eq!(transfers[0].payload.as_ref(), &[1, 2, 3]);
        assert_eq!(transfers[1].destination_offset_bytes, 16);
        assert_eq!(transfers[1].payload.as_ref(), &[9, 10]);
        assert_eq!(
            transfers
                .iter()
                .map(|transfer| transfer.payload.len())
                .sum::<usize>(),
            5,
            "sparse planning must not materialize the unwritten arena gap",
        );
    }

    #[test]
    fn sparse_program_binding_transfers_reject_overlap_and_arena_overflow() {
        let overlap = coalesce_program_binding_transfers(
            vec![
                program_binding_write(0, vec![1, 2, 3, 4]),
                program_binding_write(3, vec![5, 6]),
            ],
            8,
        );
        assert!(matches!(overlap, Err(CudaDeviceRuntimeError::Contract(_))));

        let overflow =
            coalesce_program_binding_transfers(vec![program_binding_write(7, vec![1, 2])], 8);
        assert!(matches!(overflow, Err(CudaDeviceRuntimeError::Contract(_))));
    }

    #[test]
    fn sparse_program_binding_transfers_pack_thirty_two_fixed_stride_rows() {
        let row_bytes = 80_usize;
        let stride = 131_088_u64;
        let writes = (0_u8..32)
            .map(|row| program_binding_write(128 + u64::from(row) * stride, vec![row; row_bytes]))
            .collect();
        let transfers = coalesce_program_binding_transfers(writes, 5_000_000).unwrap();

        assert_eq!(transfers.len(), 1);
        let transfer = &transfers[0];
        assert_eq!(transfer.destination_offset_bytes, 128);
        assert_eq!(transfer.destination_stride_bytes, stride);
        assert_eq!(transfer.row_bytes, row_bytes);
        assert_eq!(transfer.row_count, 32);
        assert_eq!(transfer.payload.len(), 32 * row_bytes);
        for row in 0_u8..32 {
            let start = usize::from(row) * row_bytes;
            assert!(transfer.payload[start..start + row_bytes]
                .iter()
                .all(|byte| *byte == row));
        }
    }

    #[test]
    fn sparse_program_binding_transfers_split_when_row_length_changes() {
        let transfers = coalesce_program_binding_transfers(
            vec![
                program_binding_write(0, vec![1; 8]),
                program_binding_write(64, vec![2; 8]),
                program_binding_write(128, vec![3; 16]),
                program_binding_write(256, vec![4; 16]),
                program_binding_write(384, vec![5; 8]),
            ],
            512,
        )
        .unwrap();

        assert_eq!(transfers.len(), 3);
        assert_eq!(
            transfers
                .iter()
                .map(|transfer| (transfer.row_bytes, transfer.row_count))
                .collect::<Vec<_>>(),
            vec![(8, 2), (16, 2), (8, 1)],
        );
        assert_eq!(
            transfers
                .iter()
                .map(|transfer| transfer.destination_stride_bytes)
                .collect::<Vec<_>>(),
            vec![64, 128, 8],
        );
    }

    #[test]
    fn qwen_max_context_sixty_four_patches_keep_only_live_binding_payload() {
        const PARTICIPANTS: u64 = 32;
        const RECURRENT_ROW_BYTES: u64 = 16;
        const CAUSAL_ROW_BYTES: usize = 80;
        const MAXIMUM_CONTEXT_TOKENS: u64 = 262_144;
        const PAGE_TOKENS: u64 = 16;
        const ADDRESS_BYTES: u64 = 8;
        const CONTROL_BYTES: u64 = 16;

        let causal_row_capacity =
            CONTROL_BYTES + MAXIMUM_CONTEXT_TOKENS.div_ceil(PAGE_TOKENS) * ADDRESS_BYTES;
        let recurrent_slot_capacity = PARTICIPANTS * RECURRENT_ROW_BYTES;
        let causal_slot_capacity = PARTICIPANTS * causal_row_capacity;
        let group_capacity = 3 * recurrent_slot_capacity + causal_slot_capacity;
        let arena_size = 16 * group_capacity;
        assert_eq!(arena_size, 67_141_632);

        let mut logical_patches = Vec::with_capacity(64);
        for group in 0_u64..16 {
            let group_offset = group * group_capacity;
            for recurrent in 0_u64..3 {
                let slot_offset = group_offset + recurrent * recurrent_slot_capacity;
                logical_patches.push(
                    (0_u64..PARTICIPANTS)
                        .map(|participant| {
                            program_binding_write(
                                slot_offset + participant * RECURRENT_ROW_BYTES,
                                vec![
                                    u8::try_from(recurrent).unwrap();
                                    usize::try_from(RECURRENT_ROW_BYTES).unwrap()
                                ],
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let causal_slot_offset = group_offset + 3 * recurrent_slot_capacity;
            logical_patches.push(
                (0_u64..PARTICIPANTS)
                    .map(|participant| {
                        program_binding_write(
                            causal_slot_offset + participant * causal_row_capacity,
                            vec![u8::try_from(group).unwrap(); CAUSAL_ROW_BYTES],
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(logical_patches.len(), 64);

        let transfers = coalesce_program_binding_transfers(
            logical_patches.into_iter().flatten().collect(),
            arena_size,
        )
        .unwrap();
        let live_payload_bytes = transfers
            .iter()
            .map(|transfer| transfer.payload.len())
            .sum::<usize>();
        assert_eq!(live_payload_bytes, 65_536);
        assert_eq!(transfers.len(), 32);
        assert_eq!(transfers[0].destination_offset_bytes, 0);
        assert_eq!(transfers[0].row_bytes, 1_616);
        assert_eq!(transfers[0].row_count, 1);
        assert_eq!(
            transfers[1].destination_offset_bytes,
            3 * recurrent_slot_capacity + causal_row_capacity,
        );
        assert_eq!(transfers[1].destination_stride_bytes, causal_row_capacity);
        assert_eq!(transfers[1].row_bytes, CAUSAL_ROW_BYTES);
        assert_eq!(transfers[1].row_count, 31);
        assert_eq!(transfers[2].destination_offset_bytes, group_capacity);
    }

    fn command(operation: &'static str) -> CudaDeviceCommand {
        CudaDeviceCommand {
            runtime_instance: 1,
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            executable: Some(Arc::new(CudaCommandExecutable {
                regions: Vec::new(),
                host_storage: Vec::new(),
                enqueue: Mutex::new(Box::new(|_, _, _, _| Ok(()))),
            })),
            fence_dependencies: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            reusable_execution: None,
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
            Some(&[None, Some(2)]),
            Vec::new(),
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
        assert_eq!(binding.reusable_graph_node_count(), Some(2));
        assert_eq!(binding.compute_dispatch_count(), 0);
        assert_eq!(binding.transfer_command_count(), 2);
    }

    #[test]
    fn core_logical_work_binds_node_workspace_zero_attribution() {
        let zero = || {
            CudaDeviceCommand::transfer(
                1,
                DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
                Vec::new(),
                Vec::new(),
                Box::new(|_, _, _, _| Ok(())),
            )
        };
        let error = cuda_submission_attribution(
            &[DeviceCommandPhase::Initialization],
            &[Some(0)],
            &[zero()],
            &[DeviceExecutionPath::Eager],
            Some(&[None]),
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("invalid native work metadata"));

        let logical_work =
            DeviceCommandLogicalWork::new(DeviceBatchingForm::Packed, 1, 164).unwrap();
        let bound = zero().bind_core_logical_work(logical_work).unwrap();
        let attribution = cuda_submission_attribution(
            &[DeviceCommandPhase::Initialization],
            &[Some(0)],
            &[bound],
            &[DeviceExecutionPath::Eager],
            Some(&[None]),
            Vec::new(),
        )
        .unwrap();
        let [command] = attribution.commands() else {
            panic!("expected one node workspace initialization row")
        };
        assert_eq!(command.node_index(), Some(0));
        assert_eq!(command.command_phase(), DeviceCommandPhase::Initialization);
        assert_eq!(command.batching_form(), DeviceBatchingForm::Packed);
        assert_eq!(command.participant_count(), 1);
        assert_eq!(command.token_count(), 164);
        assert_eq!(command.compute_dispatch_count(), 0);
        assert_eq!(command.transfer_command_count(), 1);
    }

    #[test]
    fn cuda_work_attribution_rejects_empty_native_work() {
        let error = command("test_invalid")
            .with_work_attribution(DeviceBatchingForm::Scalar, 1, 1, 0, 0)
            .unwrap_err();
        assert!(error.to_string().contains("no participants or native work"));
    }

    #[test]
    fn cuda_work_attribution_rejects_non_portable_native_operation_identity() {
        let invalid = command("human readable label")
            .with_work_attribution(DeviceBatchingForm::Scalar, 1, 1, 1, 0)
            .unwrap();
        let error = cuda_submission_attribution(
            &[DeviceCommandPhase::Compute],
            &[Some(0)],
            &[invalid],
            &[DeviceExecutionPath::Eager],
            Some(&[None]),
            Vec::new(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("non-portable native operation identity"));
    }
}
