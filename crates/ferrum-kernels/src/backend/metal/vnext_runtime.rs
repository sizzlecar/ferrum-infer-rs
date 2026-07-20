//! Metal implementation of the vNext device ownership boundary.
//!
//! This runtime is intentionally independent from the legacy `MetalContext`.
//! Every vNext stream owns a command queue, submissions remain asynchronous,
//! and the returned fence retains all buffers and staging storage until Metal
//! reports a quiescent terminal state.

use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::{c_void, CStr};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use ferrum_interfaces::vnext::{
    BufferDescriptor, BufferRequest, CapabilityId, CopyRegion, DefinitelyNotSubmitted,
    DeviceBufferRetention, DeviceClass, DeviceCommandBatch, DeviceDescriptor, DeviceErrorReport,
    DeviceExecutionTiming, DeviceId, DeviceRuntime, DeviceSubmissionStage,
    DeviceSubmissionTimingSink, DeviceTerminal, DeviceTerminalReceipt, DeviceTimingMeasurement,
    DeviceTimingMode, DeviceTimingUnavailableReason, DisabledDeviceSubmissionTimingSink,
    DynamicStorageProfile, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout,
    StreamState, VNextError,
};
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    BlitCommandEncoder, Buffer, BufferRef, CommandBuffer, CommandBufferRef, CommandQueue,
    ComputeCommandEncoder, ComputeCommandEncoderRef, MTLCommandBufferStatus, MTLResourceOptions,
    NSRange,
};

use super::st;

static NEXT_RUNTIME_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_STREAM_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_SUBMISSION_INSTANCE: AtomicU64 = AtomicU64::new(1);

struct MetalSubmissionStageTimer<'sink, S>
where
    S: DeviceSubmissionTimingSink,
{
    sink: &'sink S,
    stage: DeviceSubmissionStage,
    started: Option<Instant>,
}

impl<'sink, S> MetalSubmissionStageTimer<'sink, S>
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

impl<S> Drop for MetalSubmissionStageTimer<'_, S>
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

/// Typed construction input supplied by the Metal composition root.
///
/// Capability and storage profiles describe the installed provider bundle;
/// the runtime never infers them from a model name or machine memory size.
pub struct MetalDeviceRuntimeConfig {
    pub device_id: DeviceId,
    pub runtime_implementation_fingerprint: String,
    pub capabilities: BTreeSet<CapabilityId>,
    pub dynamic_storage_profiles: BTreeSet<DynamicStorageProfile>,
}

#[derive(Debug, Clone)]
pub enum MetalDeviceRuntimeError {
    Contract(String),
    CommandBuffer {
        operation: &'static str,
        status: MTLCommandBufferStatus,
        error_code: Option<i64>,
        detail: Option<String>,
    },
}

impl MetalDeviceRuntimeError {
    pub(crate) fn contract(message: impl Into<String>) -> Self {
        Self::Contract(message.into())
    }

    fn command_buffer_status(operation: &'static str, status: MTLCommandBufferStatus) -> Self {
        Self::CommandBuffer {
            operation,
            status,
            error_code: None,
            detail: None,
        }
    }

    fn command_buffer_failure(operation: &'static str, command_buffer: &CommandBufferRef) -> Self {
        let (error_code, detail) = metal_command_buffer_error(command_buffer);
        Self::CommandBuffer {
            operation,
            status: command_buffer.status(),
            error_code,
            detail,
        }
    }
}

impl fmt::Display for MetalDeviceRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(message) => formatter.write_str(message),
            Self::CommandBuffer {
                operation,
                status,
                error_code,
                detail,
            } => {
                write!(formatter, "Metal {operation} failed with status {status:?}")?;
                if let Some(error_code) = error_code {
                    write!(formatter, " and error code {error_code}")?;
                }
                if let Some(detail) = detail {
                    write!(formatter, ": {detail}")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for MetalDeviceRuntimeError {}

fn metal_command_buffer_error(command_buffer: &CommandBufferRef) -> (Option<i64>, Option<String>) {
    unsafe {
        let error: *mut Object = msg_send![command_buffer, error];
        if error.is_null() {
            return (None, None);
        }
        let code: metal::NSInteger = msg_send![error, code];
        let description: *mut Object = msg_send![error, localizedDescription];
        let detail = if description.is_null() {
            None
        } else {
            let utf8: *const std::os::raw::c_char = msg_send![description, UTF8String];
            (!utf8.is_null()).then(|| CStr::from_ptr(utf8).to_string_lossy().into_owned())
        };
        (Some(code as i64), detail)
    }
}

struct MetalAllocation {
    base: Buffer,
    aligned_offset_bytes: u64,
    requested_bytes: u64,
}

/// One core-owned Metal allocation with its exact admitted descriptor.
pub struct MetalDeviceBuffer {
    descriptor: BufferDescriptor,
    runtime_instance: u64,
    allocation: Arc<MetalAllocation>,
}

impl fmt::Debug for MetalDeviceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalDeviceBuffer")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl MetalDeviceBuffer {
    fn region(&self, range: Range<u64>) -> Result<MetalBufferRegion, MetalDeviceRuntimeError> {
        self.region_with_retention(range, None)
    }

    pub(crate) fn retained_region(
        &self,
        range: Range<u64>,
        retention: DeviceBufferRetention,
    ) -> Result<MetalBufferRegion, MetalDeviceRuntimeError> {
        self.region_with_retention(range, Some(retention))
    }

    fn region_with_retention(
        &self,
        range: Range<u64>,
        core_retention: Option<DeviceBufferRetention>,
    ) -> Result<MetalBufferRegion, MetalDeviceRuntimeError> {
        if range.start >= range.end
            || range.end > self.descriptor.size_bytes
            || range.end > self.allocation.requested_bytes
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal buffer region is empty or outside its admitted allocation",
            ));
        }
        let offset_bytes = self
            .allocation
            .aligned_offset_bytes
            .checked_add(range.start)
            .ok_or_else(|| MetalDeviceRuntimeError::contract("Metal buffer offset overflows"))?;
        Ok(MetalBufferRegion {
            allocation: Arc::clone(&self.allocation),
            _core_retention: core_retention,
            runtime_instance: self.runtime_instance,
            offset_bytes,
            length_bytes: range.end - range.start,
            element_type: self.descriptor.element_type,
        })
    }
}

/// Owned physical Metal range retained by an encoded command and its fence.
#[derive(Clone)]
pub(crate) struct MetalBufferRegion {
    allocation: Arc<MetalAllocation>,
    _core_retention: Option<DeviceBufferRetention>,
    runtime_instance: u64,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

impl MetalBufferRegion {
    pub(crate) fn buffer(&self) -> &BufferRef {
        &self.allocation.base
    }

    pub(crate) const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub(crate) const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub(crate) const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub(crate) fn same_physical_region(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
            && self.offset_bytes == other.offset_bytes
            && self.length_bytes == other.length_bytes
            && self.element_type == other.element_type
    }
}

/// Safe, submission-local encoder. It owns the encoder references retained
/// from the command buffer and closes compute before switching to blit work.
pub(crate) struct MetalSubmissionEncoder {
    command_buffer: CommandBuffer,
    compute: Option<ComputeCommandEncoder>,
}

impl MetalSubmissionEncoder {
    fn new(command_buffer: &CommandBufferRef) -> Self {
        Self {
            command_buffer: command_buffer.to_owned(),
            compute: None,
        }
    }

    pub(crate) fn compute_encoder(&mut self) -> &ComputeCommandEncoderRef {
        if self.compute.is_none() {
            self.compute = Some(self.command_buffer.new_compute_command_encoder().to_owned());
        }
        self.compute
            .as_deref()
            .expect("compute encoder was installed")
    }

    pub(crate) fn end_compute(&mut self) {
        if let Some(encoder) = self.compute.take() {
            encoder.end_encoding();
        }
    }

    pub(crate) fn with_blit<T>(
        &mut self,
        encode: impl FnOnce(&metal::BlitCommandEncoderRef) -> T,
    ) -> T {
        self.end_compute();
        let encoder =
            MetalBlitEncoderGuard(self.command_buffer.new_blit_command_encoder().to_owned());
        encode(&encoder.0)
    }

    fn finish(mut self) {
        self.end_compute();
    }
}

struct MetalBlitEncoderGuard(BlitCommandEncoder);

impl Drop for MetalBlitEncoderGuard {
    fn drop(&mut self) {
        self.0.end_encoding();
    }
}

impl Drop for MetalSubmissionEncoder {
    fn drop(&mut self) {
        self.end_compute();
    }
}

type EncodeAction = Box<
    dyn Fn(
            &mut MetalSubmissionEncoder,
            &[MetalBufferRegion],
            &[Buffer],
        ) -> Result<(), MetalDeviceRuntimeError>
        + Send
        + 'static,
>;

/// Encoded Metal work. Regions and staging buffers remain owned until the
/// exact submission fence reaches a terminal state.
pub struct MetalDeviceCommand {
    runtime_instance: u64,
    operation: &'static str,
    regions: Vec<MetalBufferRegion>,
    staging: Vec<Buffer>,
    encode: EncodeAction,
}

impl fmt::Debug for MetalDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalDeviceCommand")
            .field("runtime_instance", &self.runtime_instance)
            .field("operation", &self.operation)
            .field("region_count", &self.regions.len())
            .field("staging_count", &self.staging.len())
            .finish_non_exhaustive()
    }
}

impl MetalDeviceCommand {
    /// Backend-local providers use this after translating every logical view
    /// into retained physical regions.
    pub(crate) fn operation(
        operation: &'static str,
        regions: Vec<MetalBufferRegion>,
        encode: impl Fn(
                &mut MetalSubmissionEncoder,
                &[MetalBufferRegion],
            ) -> Result<(), MetalDeviceRuntimeError>
            + Send
            + 'static,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let runtime_instance = common_runtime_instance(&regions)?;
        Ok(Self {
            runtime_instance,
            operation,
            regions,
            staging: Vec::new(),
            encode: Box::new(move |encoder, regions, _staging| encode(encoder, regions)),
        })
    }

    fn transfer(
        runtime_instance: u64,
        operation: &'static str,
        regions: Vec<MetalBufferRegion>,
        staging: Vec<Buffer>,
        encode: EncodeAction,
    ) -> Self {
        Self {
            runtime_instance,
            operation,
            regions,
            staging,
            encode,
        }
    }

    fn encode(&self, encoder: &mut MetalSubmissionEncoder) -> Result<(), MetalDeviceRuntimeError> {
        (self.encode)(encoder, &self.regions, &self.staging)
    }
}

fn common_runtime_instance(regions: &[MetalBufferRegion]) -> Result<u64, MetalDeviceRuntimeError> {
    let runtime_instance = regions
        .first()
        .map(|region| region.runtime_instance)
        .ok_or_else(|| {
            MetalDeviceRuntimeError::contract("Metal operation has no buffer regions")
        })?;
    if regions
        .iter()
        .any(|region| region.runtime_instance != runtime_instance)
    {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal operation mixes buffers from different runtime instances",
        ));
    }
    Ok(runtime_instance)
}

struct MetalStreamState {
    recording: AtomicBool,
    failed: AtomicBool,
    in_flight: AtomicU64,
}

impl MetalStreamState {
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

    fn begin_submission(&self) -> Result<(), MetalDeviceRuntimeError> {
        if self.failed.load(Ordering::Acquire)
            || self
                .recording
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal stream is failed or already recording a submission",
            ));
        }
        if self.failed.load(Ordering::Acquire) {
            self.recording.store(false, Ordering::Release);
            return Err(MetalDeviceRuntimeError::contract("Metal stream is failed"));
        }
        Ok(())
    }

    fn cancel_recording(&self) {
        self.recording.store(false, Ordering::Release);
    }

    fn submission_recorded(&self) -> Result<(), MetalDeviceRuntimeError> {
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MetalDeviceRuntimeError::contract("Metal in-flight count overflowed"))?;
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

pub struct MetalDeviceStream {
    id: u64,
    runtime_instance: u64,
    queue: CommandQueue,
    state: Arc<MetalStreamState>,
    pending: Arc<MetalPendingSubmissions>,
}

impl fmt::Debug for MetalDeviceStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalDeviceStream")
            .field("id", &self.id)
            .field("runtime_instance", &self.runtime_instance)
            .field("state", &self.state.snapshot())
            .field("pending_count", &self.pending.len())
            .finish_non_exhaustive()
    }
}

struct MetalPendingSubmissions {
    command_buffers: Mutex<Vec<(u64, CommandBuffer)>>,
}

impl MetalPendingSubmissions {
    fn new() -> Self {
        Self {
            command_buffers: Mutex::new(Vec::new()),
        }
    }

    fn command_buffers(&self) -> MutexGuard<'_, Vec<(u64, CommandBuffer)>> {
        self.command_buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn insert(&self, submission_id: u64, command_buffer: CommandBuffer) {
        self.command_buffers().push((submission_id, command_buffer));
    }

    fn remove(&self, submission_id: u64) {
        self.command_buffers()
            .retain(|(current, _)| *current != submission_id);
    }

    fn snapshot(&self) -> Vec<CommandBuffer> {
        self.command_buffers()
            .iter()
            .map(|(_, command_buffer)| command_buffer.clone())
            .collect()
    }

    fn clear(&self) {
        self.command_buffers().clear();
    }

    fn len(&self) -> usize {
        self.command_buffers().len()
    }
}

enum MetalFenceTiming {
    NotRequested,
    Unavailable,
}

pub struct MetalDeviceFence {
    submission_id: u64,
    command_buffer: CommandBuffer,
    timing: MetalFenceTiming,
    stream_state: Arc<MetalStreamState>,
    pending: Arc<MetalPendingSubmissions>,
    terminal_accounted: AtomicBool,
    commands: Vec<MetalDeviceCommand>,
}

impl fmt::Debug for MetalDeviceFence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalDeviceFence")
            .field("status", &self.command_buffer.status())
            .field("stream_state", &self.stream_state.snapshot())
            .finish_non_exhaustive()
    }
}

impl MetalDeviceFence {
    fn mark_terminal(&self) {
        if !self.terminal_accounted.swap(true, Ordering::AcqRel) {
            self.pending.remove(self.submission_id);
            self.stream_state.finish_one();
        }
    }

    fn terminal_receipt<E>(&self, terminal: DeviceTerminal<E>) -> DeviceTerminalReceipt<E> {
        match self.timing {
            MetalFenceTiming::NotRequested => DeviceTerminalReceipt::unprofiled(terminal),
            MetalFenceTiming::Unavailable => DeviceTerminalReceipt::profiled(
                terminal,
                DeviceTimingMeasurement::<DeviceExecutionTiming>::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                ),
            ),
        }
    }

    fn failed_terminal(
        &self,
        operation: &'static str,
    ) -> DeviceTerminalReceipt<MetalDeviceRuntimeError> {
        self.stream_state.fail();
        self.mark_terminal();
        self.terminal_receipt(DeviceTerminal::FailedButQuiescent(
            MetalDeviceRuntimeError::command_buffer_failure(operation, &self.command_buffer),
        ))
    }
}

impl Drop for MetalDeviceFence {
    fn drop(&mut self) {
        if self.terminal_accounted.load(Ordering::Acquire) {
            return;
        }
        self.command_buffer.wait_until_completed();
        match self.command_buffer.status() {
            MTLCommandBufferStatus::Completed => self.mark_terminal(),
            MTLCommandBufferStatus::Error => {
                self.stream_state.fail();
                self.mark_terminal();
            }
            _ => {
                // Retain a second command-buffer reference and all command
                // ownership if Metal ever violates wait-until-completed's
                // terminal guarantee.
                self.stream_state.fail();
                std::mem::forget(self.command_buffer.clone());
                std::mem::forget(std::mem::take(&mut self.commands));
            }
        }
    }
}

/// Concrete Metal primitive runtime consumed by the shared vNext resource and
/// operation dispatch layers.
pub struct MetalDeviceRuntime {
    descriptor: DeviceDescriptor,
    runtime_instance: u64,
    device: metal::Device,
}

impl fmt::Debug for MetalDeviceRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalDeviceRuntime")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl MetalDeviceRuntime {
    pub fn new(config: MetalDeviceRuntimeConfig) -> Result<Self, MetalDeviceRuntimeError> {
        let device = st().pipes.device.clone();
        let descriptor = DeviceDescriptor {
            id: config.device_id,
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: device.recommended_max_working_set_size(),
            runtime_implementation_fingerprint: config.runtime_implementation_fingerprint,
            capabilities: config.capabilities,
            dynamic_storage_profiles: config.dynamic_storage_profiles,
        };
        descriptor
            .validate()
            .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
        let runtime_instance = NEXT_RUNTIME_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MetalDeviceRuntimeError::contract("Metal runtime identity exhausted"))?;
        Ok(Self {
            descriptor,
            runtime_instance,
            device,
        })
    }

    pub(crate) fn device(&self) -> &metal::Device {
        &self.device
    }

    fn allocate_request(
        &self,
        request: &BufferRequest,
    ) -> Result<MetalDeviceBuffer, MetalDeviceRuntimeError> {
        let extra_alignment = request.alignment_bytes().checked_sub(1).ok_or_else(|| {
            MetalDeviceRuntimeError::contract("Metal allocation alignment is zero")
        })?;
        let allocation_bytes = request
            .size_bytes()
            .checked_add(extra_alignment)
            .ok_or_else(|| MetalDeviceRuntimeError::contract("Metal allocation size overflows"))?;
        let base = self
            .device
            .new_buffer(allocation_bytes, MTLResourceOptions::StorageModeShared);
        let base_address = u64::try_from(base.contents() as usize)
            .map_err(|_| MetalDeviceRuntimeError::contract("Metal buffer address exceeds u64"))?;
        if base_address == 0 {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal shared allocation returned a null host address",
            ));
        }
        let alignment = request.alignment_bytes();
        let aligned_address = base_address
            .checked_add(alignment - 1)
            .map(|address| address & !(alignment - 1))
            .ok_or_else(|| MetalDeviceRuntimeError::contract("Metal aligned address overflows"))?;
        let aligned_offset_bytes = aligned_address - base_address;
        let admitted_end = aligned_offset_bytes
            .checked_add(request.size_bytes())
            .ok_or_else(|| MetalDeviceRuntimeError::contract("Metal admitted range overflows"))?;
        if admitted_end > base.length() {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal allocation cannot satisfy the admitted alignment",
            ));
        }
        Ok(MetalDeviceBuffer {
            descriptor: BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
            runtime_instance: self.runtime_instance,
            allocation: Arc::new(MetalAllocation {
                base,
                aligned_offset_bytes,
                requested_bytes: request.size_bytes(),
            }),
        })
    }

    fn validate_buffer(&self, buffer: &MetalDeviceBuffer) -> Result<(), MetalDeviceRuntimeError> {
        if buffer.runtime_instance != self.runtime_instance {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal buffer belongs to another runtime instance",
            ));
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &MetalDeviceStream) -> Result<(), MetalDeviceRuntimeError> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal stream belongs to another runtime instance",
            ));
        }
        Ok(())
    }

    fn submit_commands<S>(
        &self,
        stream: &mut MetalDeviceStream,
        commands: Vec<MetalDeviceCommand>,
        timing_mode: DeviceTimingMode,
        timing_sink: &S,
    ) -> Result<MetalDeviceFence, DefinitelyNotSubmitted<MetalDeviceRuntimeError>>
    where
        S: DeviceSubmissionTimingSink,
    {
        let validate_stage = MetalSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::ValidateAndPrepare,
        );
        if let Err(error) = self.validate_stream(stream) {
            return Err(DefinitelyNotSubmitted::new(error));
        }
        if commands.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                MetalDeviceRuntimeError::contract("Metal command batch is empty"),
            ));
        }
        if commands
            .iter()
            .any(|command| command.runtime_instance != self.runtime_instance)
        {
            return Err(DefinitelyNotSubmitted::new(
                MetalDeviceRuntimeError::contract(
                    "Metal command batch contains work from another runtime instance",
                ),
            ));
        }
        if let Err(error) = stream.state.begin_submission() {
            return Err(DefinitelyNotSubmitted::new(error));
        }
        drop(validate_stage);

        let begin_timing_stage =
            MetalSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::BeginTiming);
        let timing = match timing_mode {
            DeviceTimingMode::Off => MetalFenceTiming::NotRequested,
            DeviceTimingMode::Completion => MetalFenceTiming::Unavailable,
        };
        drop(begin_timing_stage);

        let command_buffer = stream.queue.new_command_buffer().to_owned();
        let mut encoder = MetalSubmissionEncoder::new(&command_buffer);
        let enqueue_stage =
            MetalSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::EnqueueCommands);
        for command in &commands {
            if let Err(error) = command.encode(&mut encoder) {
                stream.state.cancel_recording();
                return Err(DefinitelyNotSubmitted::new(error));
            }
        }
        encoder.finish();
        drop(enqueue_stage);

        let fence_stage = MetalSubmissionStageTimer::start(
            timing_sink,
            DeviceSubmissionStage::RecordFenceAndAccount,
        );
        let submission_id = match NEXT_SUBMISSION_INSTANCE.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_add(1),
        ) {
            Ok(submission_id) => submission_id,
            Err(_) => {
                stream.state.cancel_recording();
                return Err(DefinitelyNotSubmitted::new(
                    MetalDeviceRuntimeError::contract("Metal submission identity exhausted"),
                ));
            }
        };
        if let Err(error) = stream.state.submission_recorded() {
            stream.state.cancel_recording();
            return Err(DefinitelyNotSubmitted::new(error));
        }
        stream.pending.insert(submission_id, command_buffer.clone());
        command_buffer.commit();
        let fence = MetalDeviceFence {
            submission_id,
            command_buffer,
            timing,
            stream_state: Arc::clone(&stream.state),
            pending: Arc::clone(&stream.pending),
            terminal_accounted: AtomicBool::new(false),
            commands,
        };
        drop(fence_stage);
        Ok(fence)
    }
}

fn checked_usize(value: u64, context: &'static str) -> Result<usize, MetalDeviceRuntimeError> {
    usize::try_from(value).map_err(|_| {
        MetalDeviceRuntimeError::contract(format!("{context} exceeds host address space"))
    })
}

fn checked_end(
    offset: u64,
    length: u64,
    capacity: u64,
    context: &'static str,
) -> Result<u64, MetalDeviceRuntimeError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| MetalDeviceRuntimeError::contract(format!("{context} range overflows")))?;
    if length == 0 || end > capacity {
        return Err(MetalDeviceRuntimeError::contract(format!(
            "{context} range is empty or outside its buffer"
        )));
    }
    Ok(end)
}

impl DeviceRuntime for MetalDeviceRuntime {
    type Buffer = MetalDeviceBuffer;
    type Stream = MetalDeviceStream;
    type Command = MetalDeviceCommand;
    type Fence = MetalDeviceFence;
    type Error = MetalDeviceRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(
        &self,
        permit: ferrum_interfaces::vnext::DeviceAllocationPermit<'_>,
    ) -> Result<Self::Buffer, Self::Error> {
        self.allocate_request(permit.into_request())
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        let id = NEXT_STREAM_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| MetalDeviceRuntimeError::contract("Metal stream identity exhausted"))?;
        Ok(MetalDeviceStream {
            id,
            runtime_instance: self.runtime_instance,
            queue: self.device.new_command_queue(),
            state: Arc::new(MetalStreamState::new()),
            pending: Arc::new(MetalPendingSubmissions::new()),
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.runtime_instance != self.runtime_instance {
            StreamState::Failed
        } else {
            stream.state.snapshot()
        }
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
            .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
        if source.descriptor.element_type != destination.descriptor.element_type {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal copy requires matching source and destination element types",
            ));
        }
        let source_region = source.region(
            region.source_offset_bytes()..region.source_offset_bytes() + region.length_bytes(),
        )?;
        let destination_region = destination.region(
            region.destination_offset_bytes()
                ..region.destination_offset_bytes() + region.length_bytes(),
        )?;
        Ok(MetalDeviceCommand::transfer(
            self.runtime_instance,
            "device copy",
            vec![source_region, destination_region],
            Vec::new(),
            Box::new(|encoder, regions, _staging| {
                encoder.with_blit(|blit| {
                    blit.copy_from_buffer(
                        regions[0].buffer(),
                        regions[0].offset_bytes(),
                        regions[1].buffer(),
                        regions[1].offset_bytes(),
                        regions[0].length_bytes(),
                    );
                });
                Ok(())
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
            .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
        if source_layout.element_type() != destination.descriptor.element_type {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal upload layout differs from destination element type",
            ));
        }
        let source_bytes = u64::try_from(source.len())
            .map_err(|_| MetalDeviceRuntimeError::contract("Metal upload size exceeds u64"))?;
        let destination_end = checked_end(
            destination_offset_bytes,
            source_bytes,
            destination.descriptor.size_bytes,
            "Metal upload",
        )?;
        let destination_region = destination.region(destination_offset_bytes..destination_end)?;
        let staging = self.device.new_buffer_with_data(
            source.as_ptr().cast::<c_void>(),
            source_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalDeviceCommand::transfer(
            self.runtime_instance,
            "host upload",
            vec![destination_region],
            vec![staging],
            Box::new(|encoder, regions, staging| {
                encoder.with_blit(|blit| {
                    blit.copy_from_buffer(
                        &staging[0],
                        0,
                        regions[0].buffer(),
                        regions[0].offset_bytes(),
                        regions[0].length_bytes(),
                    );
                });
                Ok(())
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
            "Metal zero",
        )?;
        let destination_region = destination.region(destination_offset_bytes..destination_end)?;
        Ok(MetalDeviceCommand::transfer(
            self.runtime_instance,
            "device zero",
            vec![destination_region],
            Vec::new(),
            Box::new(|encoder, regions, _staging| {
                encoder.with_blit(|blit| {
                    blit.fill_buffer(
                        regions[0].buffer(),
                        NSRange::new(regions[0].offset_bytes(), regions[0].length_bytes()),
                        0,
                    );
                });
                Ok(())
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
        let timing_mode = commands.timing_mode();
        self.submit_commands(stream, commands.into_commands(), timing_mode, timing_sink)
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        match fence.command_buffer.status() {
            MTLCommandBufferStatus::NotEnqueued
            | MTLCommandBufferStatus::Enqueued
            | MTLCommandBufferStatus::Committed
            | MTLCommandBufferStatus::Scheduled => FenceQuery::Pending,
            MTLCommandBufferStatus::Completed => {
                fence.mark_terminal();
                FenceQuery::Terminal(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            MTLCommandBufferStatus::Error => {
                FenceQuery::Terminal(fence.failed_terminal("command buffer execution"))
            }
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        fence.command_buffer.wait_until_completed();
        match fence.command_buffer.status() {
            MTLCommandBufferStatus::Completed => {
                fence.mark_terminal();
                Ok(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            MTLCommandBufferStatus::Error => Ok(fence.failed_terminal("command buffer execution")),
            status => {
                fence.stream_state.fail();
                Err(FenceIndeterminate::new(
                    MetalDeviceRuntimeError::command_buffer_status("fence wait", status),
                ))
            }
        }
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        self.validate_stream(stream)?;
        let mut failure = None;
        for command_buffer in stream.pending.snapshot() {
            command_buffer.wait_until_completed();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                failure = Some(MetalDeviceRuntimeError::command_buffer_failure(
                    "stream synchronization",
                    &command_buffer,
                ));
            } else if command_buffer.status() != MTLCommandBufferStatus::Completed {
                stream.state.fail();
                return Err(MetalDeviceRuntimeError::command_buffer_status(
                    "stream synchronization",
                    command_buffer.status(),
                ));
            }
        }
        stream.pending.clear();
        stream.state.synchronized();
        if let Some(failure) = failure {
            stream.state.fail();
            return Err(failure);
        }
        Ok(())
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
            return Err(MetalDeviceRuntimeError::contract(
                "Metal readback layout differs from source element type",
            ));
        }
        let output_bytes = output_layout
            .byte_len()
            .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
        let source_end = checked_end(
            region.source_offset_bytes(),
            region.length_bytes(),
            source.descriptor.size_bytes,
            "Metal readback source",
        )?;
        let output_end = checked_end(
            region.destination_offset_bytes(),
            region.length_bytes(),
            output_bytes,
            "Metal readback output",
        )?;
        self.synchronize(stream)?;
        let source_region = source.region(region.source_offset_bytes()..source_end)?;
        let mut output = vec![0_u8; checked_usize(output_bytes, "Metal readback output")?];
        let output_start = checked_usize(
            region.destination_offset_bytes(),
            "Metal readback output offset",
        )?;
        let output_end = checked_usize(output_end, "Metal readback output end")?;
        let source_start =
            checked_usize(source_region.offset_bytes(), "Metal readback source offset")?;
        unsafe {
            let source_pointer = source_region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(source_start);
            std::ptr::copy_nonoverlapping(
                source_pointer,
                output[output_start..output_end].as_mut_ptr(),
                output_end - output_start,
            );
        }
        Ok(output)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        let (code, retryable) = match error {
            MetalDeviceRuntimeError::Contract(_) => ("metal_runtime_contract", false),
            MetalDeviceRuntimeError::CommandBuffer {
                error_code: Some(8),
                ..
            } => ("metal_out_of_memory", true),
            MetalDeviceRuntimeError::CommandBuffer { .. } => ("metal_command_buffer_error", false),
        };
        DeviceErrorReport::new(code, error.to_string(), retryable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        BufferUsage, DynamicStorageAllocator, DynamicStorageView, ResourceId,
    };

    fn runtime() -> MetalDeviceRuntime {
        MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
            device_id: DeviceId::new("device/metal/test").expect("device id"),
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .expect("storage profile")]),
        })
        .expect("Metal runtime")
    }

    fn buffer_request(resource: &str) -> BufferRequest {
        BufferRequest::new(
            ResourceId::new(resource).expect("resource id"),
            8,
            64,
            BufferUsage::Transfer,
            ElementType::U8,
        )
        .expect("buffer request")
    }

    #[test]
    fn ordered_transfer_batch_is_async_and_readback_is_exact() {
        let runtime = runtime();
        let source = runtime
            .allocate_request(&buffer_request("resource/source"))
            .expect("source allocation");
        let destination = runtime
            .allocate_request(&buffer_request("resource/destination"))
            .expect("destination allocation");
        let source_address = source.allocation.base.contents() as usize
            + usize::try_from(source.allocation.aligned_offset_bytes).expect("aligned offset");
        assert_eq!(source_address % 64, 0);
        let mut stream = runtime.create_stream().expect("stream");
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);

        let commands = vec![
            runtime
                .encode_upload(
                    &[1, 2, 3, 4, 5, 6, 7, 8],
                    HostTransferLayout::new(ElementType::U8, 8).expect("upload layout"),
                    &source,
                    0,
                )
                .expect("upload command"),
            runtime
                .encode_copy(
                    &source,
                    &destination,
                    CopyRegion::new(0, 0, 8).expect("copy region"),
                )
                .expect("copy command"),
            runtime
                .encode_zero(&destination, 2, 3)
                .expect("zero command"),
        ];
        let fence = runtime
            .submit_commands(
                &mut stream,
                commands,
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("submission");
        assert_eq!(runtime.stream_state(&stream), StreamState::Submitted);
        let terminal = runtime.wait_fence(&fence).expect("terminal fence");
        assert!(terminal.terminal().is_succeeded());
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(stream.pending.len(), 0);

        let output = runtime
            .readback(
                &mut stream,
                &destination,
                CopyRegion::new(0, 0, 8).expect("readback region"),
                HostTransferLayout::new(ElementType::U8, 8).expect("readback layout"),
            )
            .expect("readback");
        assert_eq!(output, [1, 2, 0, 0, 0, 6, 7, 8]);
    }

    #[test]
    fn encode_failure_is_definitely_not_submitted_and_restores_stream() {
        let runtime = runtime();
        let buffer = runtime
            .allocate_request(&buffer_request("resource/failing-operation"))
            .expect("allocation");
        let command = MetalDeviceCommand::operation(
            "failing operation",
            vec![buffer.region(0..8).expect("region")],
            |_encoder, _regions| Err(MetalDeviceRuntimeError::contract("expected encode failure")),
        )
        .expect("command");
        let mut stream = runtime.create_stream().expect("stream");

        let failure = runtime
            .submit_commands(
                &mut stream,
                vec![command],
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect_err("encoding must fail before commit");
        assert_eq!(failure.error().to_string(), "expected encode failure");
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(stream.pending.len(), 0);
    }

    #[test]
    fn command_buffer_out_of_memory_is_retryable() {
        let runtime = runtime();
        let report = runtime
            .describe_error(&MetalDeviceRuntimeError::CommandBuffer {
                operation: "test execution",
                status: MTLCommandBufferStatus::Error,
                error_code: Some(8),
                detail: Some("out of memory".to_owned()),
            })
            .expect("error report");
        assert_eq!(report.code(), "metal_out_of_memory");
        assert!(report.retryable());
    }
}
