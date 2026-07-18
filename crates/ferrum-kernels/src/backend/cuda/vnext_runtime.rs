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
    BufferDescriptor, CapabilityId, CopyRegion, DefinitelyNotSubmitted, DeviceClass,
    DeviceCommandBatch, DeviceCommandEntry, DeviceCommandPhase, DeviceDescriptor,
    DeviceErrorReport, DeviceExecutionTiming, DeviceId, DeviceRuntime, DeviceSubmissionStage,
    DeviceSubmissionTimingSink, DeviceTerminal, DeviceTerminalReceipt, DeviceTimingMeasurement,
    DeviceTimingMode, DeviceTimingUnavailableReason, DisabledDeviceSubmissionTimingSink,
    DynamicStorageProfile, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout,
    StreamState, VNextError,
};

use super::vnext_replay::{CudaCommandReplayKey, CudaExecutableCache};

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
    pub maximum_reusable_executables_per_stream: usize,
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
    pub(crate) fn region(
        &self,
        range: Range<u64>,
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
        Ok(CudaBufferRegion {
            _allocation: Arc::clone(&self.allocation),
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

pub(crate) struct CudaCommandPayload {
    regions: Vec<CudaBufferRegion>,
    host_storage: Vec<Box<[u8]>>,
    enqueue: Mutex<EnqueueAction>,
}

/// Encoded CUDA work. Buffer and host-transfer storage stays alive until the
/// returned fence reaches a terminal state.
pub struct CudaDeviceCommand {
    runtime_instance: u64,
    operation: &'static str,
    payload: Arc<CudaCommandPayload>,
    replay_key: Option<CudaCommandReplayKey>,
}

impl fmt::Debug for CudaDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaDeviceCommand")
            .field("runtime_instance", &self.runtime_instance)
            .field("operation", &self.operation)
            .field("region_count", &self.payload.regions.len())
            .field("host_storage_count", &self.payload.host_storage.len())
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
            Some(replay_key),
            enqueue,
        )
    }

    fn operation_with_host_storage_and_blas_inner(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
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
        if host_storage.iter().any(|storage| storage.is_empty()) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA operation host storage contains an empty region",
            ));
        }
        let replay_key = bind_replay_key(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            payload: Arc::new(CudaCommandPayload {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            }),
            replay_key,
        })
    }

    fn operation_inner(
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
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
        let host_storage = Vec::new();
        let replay_key = bind_replay_key(replay_key, operation, &regions, &host_storage);
        Ok(Self {
            runtime_instance,
            operation,
            payload: Arc::new(CudaCommandPayload {
                regions,
                host_storage,
                enqueue: Mutex::new(Box::new(enqueue)),
            }),
            replay_key,
        })
    }

    fn transfer(
        runtime_instance: u64,
        operation: &'static str,
        regions: Vec<CudaBufferRegion>,
        host_storage: Vec<Box<[u8]>>,
        enqueue: EnqueueAction,
    ) -> Self {
        let payload = Arc::new(CudaCommandPayload {
            regions,
            host_storage,
            enqueue: Mutex::new(enqueue),
        });
        Self {
            runtime_instance,
            operation,
            payload,
            replay_key: None,
        }
    }

    pub(crate) fn enqueue(
        &self,
        stream: &CudaStream,
        blas: &CudaBlas,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let enqueue = self
            .payload
            .enqueue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enqueue(
            stream,
            blas,
            &self.payload.regions,
            &self.payload.host_storage,
        )
    }

    pub(crate) const fn replay_key(&self) -> Option<CudaCommandReplayKey> {
        self.replay_key
    }

    pub(crate) fn payload(&self) -> Arc<CudaCommandPayload> {
        Arc::clone(&self.payload)
    }
}

fn bind_replay_key(
    replay_key: Option<CudaCommandReplayKey>,
    operation: &'static str,
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> Option<CudaCommandReplayKey> {
    replay_key.map(|key| {
        key.bind_runtime_payload(
            operation,
            regions.iter().map(|region| {
                (
                    region.device_ptr,
                    region.length_bytes,
                    region.element_type.size_bytes(),
                )
            }),
            host_storage,
        )
    })
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
        let elapsed_ms = match unsafe {
            cudarc::driver::result::event::elapsed(start.cu_event(), self.event.cu_event())
        } {
            Ok(elapsed_ms) if elapsed_ms.is_finite() && elapsed_ms >= 0.0 => elapsed_ms,
            Ok(_) | Err(_) => {
                return DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                );
            }
        };
        let elapsed_ns = f64::from(elapsed_ms) * 1_000_000.0;
        if elapsed_ns > u64::MAX as f64 {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::DurationOverflow,
            );
        }
        DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
            elapsed_ns.round() as u64,
        ))
    }

    fn terminal_receipt<E>(&self, terminal: DeviceTerminal<E>) -> DeviceTerminalReceipt<E> {
        match &self.timing {
            CudaFenceTiming::NotRequested => DeviceTerminalReceipt::unprofiled(terminal),
            CudaFenceTiming::Events { .. } | CudaFenceTiming::Unavailable => {
                DeviceTerminalReceipt::profiled(terminal, self.execution_timing())
            }
        }
    }
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
    maximum_reusable_executables_per_stream: usize,
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
        if config.maximum_reusable_executables_per_stream == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA reusable executable cache capacity must be positive",
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
            maximum_reusable_executables_per_stream: config.maximum_reusable_executables_per_stream,
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
            executable_cache: CudaExecutableCache::new(
                self.maximum_reusable_executables_per_stream,
            ),
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.runtime_instance != self.runtime_instance {
            return StreamState::Failed;
        }
        stream.state.snapshot()
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
        let (command_phases, commands): (Vec<_>, Vec<_>) = commands
            .into_entries()
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .unzip();
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
        if let Err(error) = self.context.bind_to_thread() {
            return Err(DefinitelyNotSubmitted::new(CudaDeviceRuntimeError::driver(
                "submission context binding",
                error,
            )));
        }
        let stream_was_quiescent = stream.state.is_quiescent();
        if let Err(error) = stream.state.begin_submission() {
            return Err(DefinitelyNotSubmitted::new(error));
        }
        let mut segment_start = 0;
        while segment_start < commands.len() {
            let Some(segment_end) =
                replayable_segment_end(&command_phases, &commands, segment_start)
            else {
                segment_start += 1;
                continue;
            };
            if let Err(error) = stream.executable_cache.prepare(
                &self.context,
                &stream.stream,
                &stream.blas,
                &commands[segment_start..segment_end],
                stream_was_quiescent,
            ) {
                if !error.eager_fallback_safe() {
                    stream.state.fail();
                    self.quarantine(stream, commands);
                    panic!(
                        "CUDA submission became indeterminate while preparing a reusable executable: {error}"
                    );
                }
                tracing::debug!(
                    error = %error,
                    command_count = segment_end - segment_start,
                    "CUDA reusable executable capture rejected; using eager fallback"
                );
            }
            segment_start = segment_end;
        }
        drop(validate_stage);

        let begin_timing_stage =
            CudaSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::BeginTiming);
        let timing = match timing_mode {
            DeviceTimingMode::Off => CudaFenceTiming::NotRequested,
            DeviceTimingMode::Completion => {
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
        let mut index = 0;
        while index < commands.len() {
            let replayed_end =
                replayable_segment_end(&command_phases, &commands, index).and_then(|segment_end| {
                    match stream
                        .executable_cache
                        .launch(&stream.stream, &commands[index..segment_end])
                    {
                        Ok(true) => Some(Ok(segment_end)),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                });
            match replayed_end {
                Some(Ok(segment_end)) => {
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
            if let Err(error) = commands[index].enqueue(&stream.stream, &stream.blas) {
                stream.state.fail();
                self.quarantine(stream, commands);
                panic!("CUDA submission became indeterminate while enqueueing its batch: {error}");
            }
            index += 1;
        }
        drop(enqueue_stage);

        let fence_stage = CudaSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::RecordFenceAndAccount,
        );
        let fence_flags = match timing_mode {
            DeviceTimingMode::Off => None,
            DeviceTimingMode::Completion => {
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
            stream_state: Arc::clone(&stream.state),
            terminal_accounted: AtomicBool::new(false),
            _stream: Arc::clone(&stream.stream),
            _blas: Arc::clone(&stream.blas),
            _commands: commands,
        };
        drop(fence_stage);
        Ok(fence)
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

fn replayable_segment_end(
    phases: &[DeviceCommandPhase],
    commands: &[CudaDeviceCommand],
    start: usize,
) -> Option<usize> {
    if phases.get(start) != Some(&DeviceCommandPhase::Compute)
        || commands.get(start)?.replay_key().is_none()
    {
        return None;
    }
    let mut end = start + 1;
    while end < commands.len()
        && phases.get(end) == Some(&DeviceCommandPhase::Compute)
        && commands[end].replay_key().is_some()
    {
        end += 1;
    }
    Some(end)
}
