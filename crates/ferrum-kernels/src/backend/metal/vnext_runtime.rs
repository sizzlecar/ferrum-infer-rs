//! Metal implementation of the vNext device ownership boundary.
//!
//! This runtime is intentionally independent from the legacy `MetalContext`.
//! Every vNext stream owns a command queue, submissions remain asynchronous,
//! and the returned fence retains all buffers and staging storage until Metal
//! reports a quiescent terminal state.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{c_void, CStr};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use ferrum_interfaces::vnext::{
    BufferDescriptor, BufferRequest, BufferUsage, CapabilityId, CopyRegion, DefinitelyNotSubmitted,
    DeviceBatchingForm, DeviceBufferRetention, DeviceClass, DeviceCommandBatch, DeviceCommandEntry,
    DeviceCommandExecutionTiming, DeviceCommandPhase, DeviceDescriptor, DeviceErrorReport,
    DeviceExecutionInterval, DeviceExecutionIntervalKind, DeviceExecutionPath,
    DeviceExecutionTiming, DeviceId, DeviceNativeWorkAttribution, DeviceRuntime,
    DeviceSubmissionAttribution, DeviceSubmissionExecutionTiming, DeviceSubmissionStage,
    DeviceSubmissionTimingSink, DeviceTerminal, DeviceTerminalReceipt, DeviceTimingMeasurement,
    DeviceTimingMode, DeviceTimingUnavailableReason, DisabledDeviceSubmissionTimingSink,
    DynamicStorageProfile, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout,
    RetainedHostMemoryRegion, StaticWeightImportSession, StreamState, VNextError,
    WeightComponentPayload,
};
use metal::foreign_types::ForeignType;
use metal::objc::runtime::{Object, BOOL, YES};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    BlitCommandEncoder, BlitPassDescriptor, Buffer, BufferRef, CommandBuffer, CommandBufferRef,
    CommandQueue, ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePassDescriptor,
    CounterSampleBuffer, CounterSampleBufferDescriptor, CounterSet, MTLBuffer,
    MTLCommandBufferStatus, MTLCounterSampleBuffer, MTLCounterSamplingPoint, MTLResourceOptions,
    MTLStorageMode, NSRange,
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
    // The Metal handle must be released before its no-copy host allocation.
    _retained_host_memory: Option<RetainedHostMemoryRegion>,
}

struct MetalStaticWeightSegment {
    logical_start_bytes: u64,
    logical_end_bytes: u64,
    allocation: Arc<MetalAllocation>,
}

struct MetalStaticWeightResidency {
    set: Option<super::residency::MetalResidencySet>,
}

struct MetalStaticWeightBacking {
    // End residency before releasing the segment buffers and their mmap owners.
    _residency: Arc<MetalStaticWeightResidency>,
    segments: Box<[MetalStaticWeightSegment]>,
}

struct MetalStaticWeightArena {
    sealed: OnceLock<MetalStaticWeightBacking>,
}

enum MetalDeviceBufferBacking {
    Contiguous(Arc<MetalAllocation>),
    StaticWeights(Arc<MetalStaticWeightArena>),
}

/// One core-owned Metal allocation with its exact admitted descriptor.
pub struct MetalDeviceBuffer {
    descriptor: BufferDescriptor,
    runtime_instance: u64,
    backing: MetalDeviceBufferBacking,
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
    #[cfg(test)]
    fn contiguous_allocation(&self) -> &Arc<MetalAllocation> {
        match &self.backing {
            MetalDeviceBufferBacking::Contiguous(allocation) => allocation,
            MetalDeviceBufferBacking::StaticWeights(_) => {
                panic!("test requested a contiguous allocation from a static-weight arena")
            }
        }
    }

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
        if range.start >= range.end || range.end > self.descriptor.size_bytes {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal buffer region is empty or outside its admitted allocation",
            ));
        }
        let (allocation, logical_start_bytes) = match &self.backing {
            MetalDeviceBufferBacking::Contiguous(allocation) => {
                if range.end > allocation.requested_bytes {
                    return Err(MetalDeviceRuntimeError::contract(
                        "Metal buffer region exceeds its physical allocation",
                    ));
                }
                (Arc::clone(allocation), 0)
            }
            MetalDeviceBufferBacking::StaticWeights(arena) => {
                let backing = arena.sealed.get().ok_or_else(|| {
                    MetalDeviceRuntimeError::contract(
                        "Metal static-weight arena is not sealed for execution",
                    )
                })?;
                let position = backing
                    .segments
                    .partition_point(|segment| segment.logical_start_bytes <= range.start);
                let segment = position
                    .checked_sub(1)
                    .and_then(|index| backing.segments.get(index))
                    .filter(|segment| {
                        range.start >= segment.logical_start_bytes
                            && range.end <= segment.logical_end_bytes
                    })
                    .ok_or_else(|| {
                        MetalDeviceRuntimeError::contract(
                            "Metal static-weight range crosses or misses an imported component",
                        )
                    })?;
                (Arc::clone(&segment.allocation), segment.logical_start_bytes)
            }
        };
        let relative_offset = range.start - logical_start_bytes;
        let offset_bytes = allocation
            .aligned_offset_bytes
            .checked_add(relative_offset)
            .ok_or_else(|| MetalDeviceRuntimeError::contract("Metal buffer offset overflows"))?;
        Ok(MetalBufferRegion {
            allocation,
            _core_retention: core_retention,
            runtime_instance: self.runtime_instance,
            offset_bytes,
            length_bytes: range.end - range.start,
            element_type: self.descriptor.element_type,
        })
    }
}

struct PendingMetalStaticWeightArena {
    arena: Arc<MetalStaticWeightArena>,
    segments: Vec<MetalStaticWeightSegment>,
}

struct MetalStaticWeightImport<'runtime> {
    device: metal::Device,
    runtime_instance: u64,
    _permit: MutexGuard<'runtime, ()>,
    residency: Arc<MetalStaticWeightResidency>,
    arenas: BTreeMap<usize, PendingMetalStaticWeightArena>,
}

impl<'runtime> MetalStaticWeightImport<'runtime> {
    fn new(
        device: metal::Device,
        runtime_instance: u64,
        permit: MutexGuard<'runtime, ()>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let set = super::residency::MetalResidencySet::new(&device)
            .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
        Ok(Self {
            device,
            runtime_instance,
            _permit: permit,
            residency: Arc::new(MetalStaticWeightResidency { set }),
            arenas: BTreeMap::new(),
        })
    }
}

impl StaticWeightImportSession<MetalDeviceBuffer, MetalDeviceRuntimeError>
    for MetalStaticWeightImport<'_>
{
    fn import_component(
        &mut self,
        payload: &WeightComponentPayload<'_>,
        destination: &MetalDeviceBuffer,
        destination_offset_bytes: u64,
    ) -> Result<(), MetalDeviceRuntimeError> {
        if destination.runtime_instance != self.runtime_instance
            || destination.descriptor.usage != BufferUsage::Weights
            || destination.descriptor.element_type != payload.element_type()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static-weight import destination differs from the runtime or payload",
            ));
        }
        let length_bytes = u64::try_from(payload.bytes().len()).map_err(|_| {
            MetalDeviceRuntimeError::contract("Metal static-weight payload exceeds u64")
        })?;
        if !destination_offset_bytes.is_multiple_of(destination.descriptor.alignment_bytes) {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static-weight component offset violates its admitted alignment",
            ));
        }
        let logical_end_bytes = checked_end(
            destination_offset_bytes,
            length_bytes,
            destination.descriptor.size_bytes,
            "Metal static-weight import",
        )?;
        let MetalDeviceBufferBacking::StaticWeights(arena) = &destination.backing else {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static-weight import requires a logical weight arena",
            ));
        };
        if arena.sealed.get().is_some() {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static-weight arena is already sealed",
            ));
        }
        let allocation = metal_static_weight_allocation(
            &self.device,
            payload,
            destination.descriptor.alignment_bytes,
        )?;
        let key = Arc::as_ptr(arena) as usize;
        self.arenas
            .entry(key)
            .or_insert_with(|| PendingMetalStaticWeightArena {
                arena: Arc::clone(arena),
                segments: Vec::new(),
            })
            .segments
            .push(MetalStaticWeightSegment {
                logical_start_bytes: destination_offset_bytes,
                logical_end_bytes,
                allocation,
            });
        Ok(())
    }

    fn seal(mut self: Box<Self>) -> Result<(), MetalDeviceRuntimeError> {
        if self.arenas.is_empty() {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static-weight import contains no arenas",
            ));
        }
        let mut prepared = Vec::with_capacity(self.arenas.len());
        for (_, mut pending) in std::mem::take(&mut self.arenas) {
            if pending.arena.sealed.get().is_some() || pending.segments.is_empty() {
                return Err(MetalDeviceRuntimeError::contract(
                    "Metal static-weight arena is already sealed or empty",
                ));
            }
            pending
                .segments
                .sort_by_key(|segment| segment.logical_start_bytes);
            if pending
                .segments
                .windows(2)
                .any(|pair| pair[0].logical_end_bytes > pair[1].logical_start_bytes)
            {
                return Err(MetalDeviceRuntimeError::contract(
                    "Metal static-weight component ranges overlap",
                ));
            }
            prepared.push((pending.arena, pending.segments.into_boxed_slice()));
        }

        if let Some(set) = &self.residency.set {
            for (_, segments) in &prepared {
                for segment in segments.iter() {
                    set.add_allocation(&segment.allocation.base);
                }
            }
            let _ = set.commit_and_request();
        }
        for (arena, segments) in prepared {
            let backing = MetalStaticWeightBacking {
                _residency: Arc::clone(&self.residency),
                segments,
            };
            if arena.sealed.set(backing).is_err() {
                unreachable!("Metal static-weight publication is single-threaded and prevalidated");
            }
        }
        Ok(())
    }
}

fn metal_static_weight_allocation(
    device: &metal::DeviceRef,
    payload: &WeightComponentPayload<'_>,
    required_alignment_bytes: u64,
) -> Result<Arc<MetalAllocation>, MetalDeviceRuntimeError> {
    let required_alignment = usize::try_from(required_alignment_bytes).map_err(|_| {
        MetalDeviceRuntimeError::contract("Metal static-weight alignment exceeds usize")
    })?;
    if required_alignment == 0 || !required_alignment.is_power_of_two() {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal static-weight alignment is not a non-zero power of two",
        ));
    }
    if let Some(retained) = payload.retained_host_memory() {
        let page_size = system_page_size()?;
        let owner = retained.owner_bytes();
        let owner_start = owner.as_ptr() as usize;
        let owner_end = owner_start.checked_add(owner.len());
        let region_start = retained.bytes().as_ptr() as usize;
        let region_end = region_start.checked_add(retained.length_bytes());
        if owner_start.is_multiple_of(page_size) && region_start.is_multiple_of(required_alignment)
        {
            if let (Some(owner_end), Some(region_end)) = (owner_end, region_end) {
                let aligned_start = region_start / page_size * page_size;
                let aligned_end = region_end
                    .checked_add(page_size - 1)
                    .map(|end| end / page_size * page_size);
                if let Some(aligned_end) = aligned_end {
                    if aligned_start >= owner_start && aligned_end <= owner_end {
                        let aligned_len = aligned_end - aligned_start;
                        let base = device.new_buffer_with_bytes_no_copy(
                            aligned_start as *const c_void,
                            aligned_len as u64,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        );
                        if base.length() == aligned_len as u64 {
                            return Ok(Arc::new(MetalAllocation {
                                base,
                                aligned_offset_bytes: (region_start - aligned_start) as u64,
                                requested_bytes: retained.length_bytes() as u64,
                                _retained_host_memory: Some(retained.clone()),
                            }));
                        }
                    }
                }
            }
        }
    }

    let bytes = payload.bytes();
    let length_bytes = u64::try_from(bytes.len()).map_err(|_| {
        MetalDeviceRuntimeError::contract("Metal static-weight payload exceeds u64")
    })?;
    let extra_alignment = required_alignment_bytes - 1;
    let allocation_bytes = length_bytes.checked_add(extra_alignment).ok_or_else(|| {
        MetalDeviceRuntimeError::contract("Metal static-weight allocation size overflows")
    })?;
    let base = device.new_buffer(allocation_bytes, MTLResourceOptions::StorageModeShared);
    let base_address = base.contents() as usize;
    if base_address == 0 {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal static-weight allocation returned a null host address",
        ));
    }
    let aligned_address = base_address
        .checked_add(required_alignment - 1)
        .map(|address| address & !(required_alignment - 1))
        .ok_or_else(|| {
            MetalDeviceRuntimeError::contract("Metal static-weight aligned address overflows")
        })?;
    let aligned_offset_bytes = u64::try_from(aligned_address - base_address).map_err(|_| {
        MetalDeviceRuntimeError::contract("Metal static-weight aligned offset exceeds u64")
    })?;
    let admitted_end = aligned_offset_bytes
        .checked_add(length_bytes)
        .ok_or_else(|| {
            MetalDeviceRuntimeError::contract("Metal static-weight admitted range overflows")
        })?;
    if admitted_end > base.length() {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal static-weight allocation cannot satisfy admitted alignment",
        ));
    }
    let aligned_offset = usize::try_from(aligned_offset_bytes).map_err(|_| {
        MetalDeviceRuntimeError::contract("Metal static-weight aligned offset exceeds usize")
    })?;
    // SAFETY: the admitted range was checked against the shared Metal buffer,
    // and the source payload remains borrowed for the duration of this copy.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            base.contents().cast::<u8>().add(aligned_offset),
            bytes.len(),
        );
    }
    Ok(Arc::new(MetalAllocation {
        base,
        aligned_offset_bytes,
        requested_bytes: length_bytes,
        _retained_host_memory: None,
    }))
}

fn system_page_size() -> Result<usize, MetalDeviceRuntimeError> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = usize::try_from(page_size).map_err(|_| {
        MetalDeviceRuntimeError::contract("host page size is unavailable for Metal no-copy import")
    })?;
    if !page_size.is_power_of_two() {
        return Err(MetalDeviceRuntimeError::contract(
            "host page size is not a power of two",
        ));
    }
    Ok(page_size)
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

const METAL_COUNTER_SAMPLES_PER_PAGE: u64 = 256;
const METAL_COUNTER_MAX_PAGES: usize = 16;
const METAL_COUNTER_ERROR_VALUE: u64 = u64::MAX;

#[derive(Clone)]
enum MetalTimestampCounterSupport {
    Supported(CounterSet),
    Unsupported,
}

fn timestamp_counter_support(device: &metal::DeviceRef) -> MetalTimestampCounterSupport {
    let supports_selector = sel!(supportsCounterSampling:);
    let counter_sets_selector = sel!(counterSets);
    let timestamps_selector = sel!(sampleTimestamps:gpuTimestamp:);
    let selectors_available = unsafe {
        let supports: BOOL = msg_send![device, respondsToSelector: supports_selector];
        let counter_sets: BOOL = msg_send![device, respondsToSelector: counter_sets_selector];
        let timestamps: BOOL = msg_send![device, respondsToSelector: timestamps_selector];
        supports == YES && counter_sets == YES && timestamps == YES
    };
    if !selectors_available
        || !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary)
    {
        return MetalTimestampCounterSupport::Unsupported;
    }
    device
        .counter_sets()
        .into_iter()
        .find(|counter_set| counter_set.name() == "timestamp")
        .map_or(
            MetalTimestampCounterSupport::Unsupported,
            MetalTimestampCounterSupport::Supported,
        )
}

fn new_counter_sample_buffer_checked(
    device: &metal::DeviceRef,
    descriptor: &metal::CounterSampleBufferDescriptorRef,
) -> Result<CounterSampleBuffer, ()> {
    let mut error: *mut Object = std::ptr::null_mut();
    let buffer: *mut MTLCounterSampleBuffer = unsafe {
        msg_send![device, newCounterSampleBufferWithDescriptor: descriptor error: &mut error]
    };
    if buffer.is_null() {
        Err(())
    } else {
        Ok(unsafe { CounterSampleBuffer::from_ptr(buffer) })
    }
}

fn new_shared_buffer_checked(device: &metal::DeviceRef, length: u64) -> Result<Buffer, ()> {
    let buffer: *mut MTLBuffer = unsafe {
        msg_send![device,
            newBufferWithLength: length
            options: MTLResourceOptions::StorageModeShared
        ]
    };
    if buffer.is_null() {
        Err(())
    } else {
        Ok(unsafe { Buffer::from_ptr(buffer) })
    }
}

struct MetalCounterPageBuilder {
    sample_buffer: CounterSampleBuffer,
    resolved: Buffer,
    used_samples: u64,
}

struct MetalCounterReservation {
    sample_buffer: CounterSampleBuffer,
    start_sample_index: u64,
    end_sample_index: u64,
}

struct MetalCounterIntervalMapping {
    command_index: u32,
    kind: DeviceExecutionIntervalKind,
    subwork_id: Option<&'static str>,
    page_index: usize,
    start_sample_index: u64,
    end_sample_index: u64,
}

struct MetalCounterCaptureBuilder {
    device: metal::Device,
    counter_set: CounterSet,
    pages: Vec<MetalCounterPageBuilder>,
    mappings: Vec<MetalCounterIntervalMapping>,
    cpu_anchor_start: u64,
    gpu_anchor_start: u64,
    unavailable: bool,
}

impl MetalCounterCaptureBuilder {
    fn new(device: metal::Device, counter_set: CounterSet) -> Self {
        let mut cpu_anchor_start = 0;
        let mut gpu_anchor_start = 0;
        device.sample_timestamps(&mut cpu_anchor_start, &mut gpu_anchor_start);
        Self {
            device,
            counter_set,
            pages: Vec::new(),
            mappings: Vec::new(),
            cpu_anchor_start,
            gpu_anchor_start,
            unavailable: cpu_anchor_start == 0 || gpu_anchor_start == 0,
        }
    }

    fn add_page(&mut self) -> Result<(), ()> {
        if self.pages.len() >= METAL_COUNTER_MAX_PAGES {
            return Err(());
        }
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&self.counter_set);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(METAL_COUNTER_SAMPLES_PER_PAGE);
        descriptor.set_label("ferrum.vnext.command_timestamps");
        let sample_buffer = new_counter_sample_buffer_checked(&self.device, &descriptor)?;
        let byte_len = METAL_COUNTER_SAMPLES_PER_PAGE
            .checked_mul(std::mem::size_of::<u64>() as u64)
            .ok_or(())?;
        let resolved = new_shared_buffer_checked(&self.device, byte_len)?;
        self.pages.push(MetalCounterPageBuilder {
            sample_buffer,
            resolved,
            used_samples: 0,
        });
        Ok(())
    }

    fn reserve(
        &mut self,
        command_index: u32,
        kind: DeviceExecutionIntervalKind,
        subwork_id: Option<&'static str>,
    ) -> Option<MetalCounterReservation> {
        if self.unavailable {
            return None;
        }
        if self
            .pages
            .last()
            .is_none_or(|page| page.used_samples + 2 > METAL_COUNTER_SAMPLES_PER_PAGE)
            && self.add_page().is_err()
        {
            self.unavailable = true;
            return None;
        }
        let page_index = self.pages.len() - 1;
        let page = &mut self.pages[page_index];
        let start_sample_index = page.used_samples;
        let end_sample_index = start_sample_index + 1;
        page.used_samples += 2;
        self.mappings.push(MetalCounterIntervalMapping {
            command_index,
            kind,
            subwork_id,
            page_index,
            start_sample_index,
            end_sample_index,
        });
        Some(MetalCounterReservation {
            sample_buffer: page.sample_buffer.clone(),
            start_sample_index,
            end_sample_index,
        })
    }

    fn mark_unavailable(&mut self) {
        self.unavailable = true;
    }

    fn finish(mut self, command_buffer: &CommandBufferRef) -> MetalCounterCapture {
        if !self.unavailable && !self.pages.is_empty() {
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("ferrum.vnext.resolve_command_timestamps");
            for page in &self.pages {
                blit.resolve_counters(
                    &page.sample_buffer,
                    NSRange::new(0, page.used_samples),
                    &page.resolved,
                    0,
                );
            }
            blit.end_encoding();
        }
        let resolved_pages = self
            .pages
            .drain(..)
            .map(|page| MetalCounterPage {
                _sample_buffer: page.sample_buffer,
                resolved: page.resolved,
                used_samples: page.used_samples,
            })
            .collect();
        MetalCounterCapture {
            device: self.device,
            pages: resolved_pages,
            mappings: self.mappings.into_boxed_slice(),
            cpu_anchor_start: self.cpu_anchor_start,
            gpu_anchor_start: self.gpu_anchor_start,
            unavailable: self.unavailable,
        }
    }
}

struct MetalCounterPage {
    _sample_buffer: CounterSampleBuffer,
    resolved: Buffer,
    used_samples: u64,
}

struct MetalCounterCapture {
    device: metal::Device,
    pages: Vec<MetalCounterPage>,
    mappings: Box<[MetalCounterIntervalMapping]>,
    cpu_anchor_start: u64,
    gpu_anchor_start: u64,
    unavailable: bool,
}

impl MetalCounterCapture {
    fn resolve(&self) -> DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        if self.unavailable || self.mappings.is_empty() {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        }
        let mut cpu_anchor_end = 0;
        let mut gpu_anchor_end = 0;
        self.device
            .sample_timestamps(&mut cpu_anchor_end, &mut gpu_anchor_end);
        let Some(cpu_span) = cpu_anchor_end.checked_sub(self.cpu_anchor_start) else {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        };
        let Some(gpu_span) = gpu_anchor_end.checked_sub(self.gpu_anchor_start) else {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        };
        if cpu_span == 0 || gpu_span == 0 {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        }
        let mut page_samples = Vec::with_capacity(self.pages.len());
        for page in &self.pages {
            let Some(sample_count) = usize::try_from(page.used_samples).ok() else {
                return DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::DurationOverflow,
                );
            };
            if page.resolved.contents().is_null() {
                return DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                );
            }
            let samples = unsafe {
                std::slice::from_raw_parts(page.resolved.contents().cast::<u64>(), sample_count)
            };
            page_samples.push(samples);
        }
        let raw_interval = |mapping: &MetalCounterIntervalMapping| -> Option<(u64, u64)> {
            let samples = *page_samples.get(mapping.page_index)?;
            let start = *samples.get(usize::try_from(mapping.start_sample_index).ok()?)?;
            let end = *samples.get(usize::try_from(mapping.end_sample_index).ok()?)?;
            (start != METAL_COUNTER_ERROR_VALUE
                && end != METAL_COUNTER_ERROR_VALUE
                && start >= self.gpu_anchor_start
                && end > start
                && end <= gpu_anchor_end)
                .then_some((start, end))
        };
        let Some(origin) = self
            .mappings
            .iter()
            .filter_map(|mapping| raw_interval(mapping).map(|(start, _)| start))
            .min()
        else {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        };
        let convert = |timestamp: u64| -> Option<u64> {
            let delta = u128::from(timestamp.checked_sub(origin)?);
            let numerator = delta.checked_mul(u128::from(cpu_span))?;
            let rounded = numerator.checked_add(u128::from(gpu_span) / 2)?;
            u64::try_from(rounded / u128::from(gpu_span)).ok()
        };
        let mut commands = BTreeMap::<u32, Vec<DeviceExecutionInterval>>::new();
        for mapping in self.mappings.iter() {
            let Some((start, end)) = raw_interval(mapping) else {
                return DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                );
            };
            let Some(interval) = convert(start).zip(convert(end)).and_then(|(start, end)| {
                mapping.subwork_id.map_or_else(
                    || DeviceExecutionInterval::new(mapping.kind, start, end),
                    |subwork_id| {
                        DeviceExecutionInterval::new_labeled(mapping.kind, start, end, subwork_id)
                    },
                )
            }) else {
                return DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::DurationOverflow,
                );
            };
            commands
                .entry(mapping.command_index)
                .or_default()
                .push(interval);
        }
        let Some(commands) = commands
            .into_iter()
            .map(|(command_index, intervals)| {
                DeviceCommandExecutionTiming::new(command_index, intervals)
            })
            .collect::<Option<Vec<_>>>()
            .and_then(DeviceSubmissionExecutionTiming::new)
        else {
            return DeviceTimingMeasurement::Unavailable(
                DeviceTimingUnavailableReason::BackendMeasurementFailed,
            );
        };
        DeviceTimingMeasurement::Measured(commands)
    }
}

enum MetalFenceCommandTiming {
    NotRequested,
    Unavailable(DeviceTimingUnavailableReason),
    Captured {
        capture: MetalCounterCapture,
        resolved: OnceLock<DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>>,
    },
}

impl MetalFenceCommandTiming {
    fn measurement(
        &self,
        command_buffer: &CommandBufferRef,
    ) -> DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return match self {
                Self::NotRequested => DeviceTimingMeasurement::NotRequested,
                Self::Unavailable(reason) => DeviceTimingMeasurement::Unavailable(*reason),
                Self::Captured { .. } => DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed,
                ),
            };
        }
        match self {
            Self::NotRequested => DeviceTimingMeasurement::NotRequested,
            Self::Unavailable(reason) => DeviceTimingMeasurement::Unavailable(*reason),
            Self::Captured { capture, resolved } => {
                resolved.get_or_init(|| capture.resolve()).clone()
            }
        }
    }
}

/// Safe, submission-local encoder. It owns the encoder references retained
/// from the command buffer and closes compute before switching to blit work.
pub(crate) struct MetalSubmissionEncoder {
    command_buffer: CommandBuffer,
    compute: Option<ComputeCommandEncoder>,
    profile_enabled: bool,
    current_command_index: Option<u32>,
    command_label: Option<&'static str>,
    compute_subwork_id: Option<&'static str>,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    counter_capture: Option<MetalCounterCaptureBuilder>,
}

impl MetalSubmissionEncoder {
    fn new(
        command_buffer: &CommandBufferRef,
        profile_enabled: bool,
        counter_capture: Option<MetalCounterCaptureBuilder>,
    ) -> Self {
        Self {
            command_buffer: command_buffer.to_owned(),
            compute: None,
            profile_enabled,
            current_command_index: None,
            command_label: None,
            compute_subwork_id: None,
            compute_dispatch_count: 0,
            transfer_command_count: 0,
            counter_capture,
        }
    }

    pub(crate) fn record_compute_dispatches(&mut self, count: u64) {
        if self.profile_enabled {
            self.compute_dispatch_count = self.compute_dispatch_count.saturating_add(count);
        }
    }

    fn begin_command(&mut self, command_index: u32, command_label: &'static str) {
        if self.profile_enabled {
            // Physical-span profiling gives every core command an encoder
            // boundary. Full profiling additionally maps those intervals to
            // native operations; replay profiling retains only physical time.
            self.end_compute();
            self.current_command_index = Some(command_index);
            self.command_label = Some(command_label);
            self.compute_subwork_id = None;
        }
        self.compute_dispatch_count = 0;
        self.transfer_command_count = 0;
    }

    fn command_counts(&self) -> (u64, u64) {
        (self.compute_dispatch_count, self.transfer_command_count)
    }

    pub(crate) fn begin_compute_subwork(&mut self, subwork_id: &'static str) {
        if self.profile_enabled && !subwork_id.is_empty() {
            self.end_compute();
            self.command_label = Some(subwork_id);
            self.compute_subwork_id = Some(subwork_id);
        }
    }

    pub(crate) fn compute_encoder(&mut self) -> &ComputeCommandEncoderRef {
        if self.compute.is_none() {
            let reservation = self.counter_capture.as_mut().and_then(|capture| {
                self.current_command_index.and_then(|command_index| {
                    capture.reserve(
                        command_index,
                        DeviceExecutionIntervalKind::Compute,
                        self.compute_subwork_id,
                    )
                })
            });
            let encoder = if let Some(reservation) = reservation {
                let descriptor = ComputePassDescriptor::new();
                if let Some(attachment) = descriptor.sample_buffer_attachments().object_at(0) {
                    attachment.set_sample_buffer(&reservation.sample_buffer);
                    attachment.set_start_of_encoder_sample_index(reservation.start_sample_index);
                    attachment.set_end_of_encoder_sample_index(reservation.end_sample_index);
                    self.command_buffer
                        .compute_command_encoder_with_descriptor(descriptor)
                        .to_owned()
                } else {
                    if let Some(capture) = self.counter_capture.as_mut() {
                        capture.mark_unavailable();
                    }
                    self.command_buffer.new_compute_command_encoder().to_owned()
                }
            } else {
                self.command_buffer.new_compute_command_encoder().to_owned()
            };
            if let Some(label) = self.command_label {
                encoder.set_label(label);
            }
            self.compute = Some(encoder);
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
        self.with_blit_commands(1, encode)
    }

    pub(crate) fn with_blit_commands<T>(
        &mut self,
        command_count: u64,
        encode: impl FnOnce(&metal::BlitCommandEncoderRef) -> T,
    ) -> T {
        debug_assert!(command_count > 0);
        if self.profile_enabled {
            self.transfer_command_count = self.transfer_command_count.saturating_add(command_count);
        }
        self.end_compute();
        let reservation = self.counter_capture.as_mut().and_then(|capture| {
            self.current_command_index.and_then(|command_index| {
                capture.reserve(
                    command_index,
                    DeviceExecutionIntervalKind::Transfer,
                    self.compute_subwork_id,
                )
            })
        });
        let encoder = if let Some(reservation) = reservation {
            let descriptor = BlitPassDescriptor::new();
            if let Some(attachment) = descriptor.sample_buffer_attachments().object_at(0) {
                attachment.set_sample_buffer(&reservation.sample_buffer);
                attachment.set_start_of_encoder_sample_index(reservation.start_sample_index);
                attachment.set_end_of_encoder_sample_index(reservation.end_sample_index);
                self.command_buffer
                    .blit_command_encoder_with_descriptor(descriptor)
                    .to_owned()
            } else {
                if let Some(capture) = self.counter_capture.as_mut() {
                    capture.mark_unavailable();
                }
                self.command_buffer.new_blit_command_encoder().to_owned()
            }
        } else {
            self.command_buffer.new_blit_command_encoder().to_owned()
        };
        if let Some(label) = self.command_label {
            encoder.set_label(label);
        }
        let encoder = MetalBlitEncoderGuard(encoder);
        encode(&encoder.0)
    }

    fn finish(mut self) -> Option<MetalCounterCapture> {
        self.end_compute();
        self.counter_capture
            .take()
            .map(|capture| capture.finish(&self.command_buffer))
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
    batching_form: DeviceBatchingForm,
    participant_count: u32,
    token_count: u64,
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
            .field("batching_form", &self.batching_form)
            .field("participant_count", &self.participant_count)
            .field("token_count", &self.token_count)
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
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 1,
            token_count: 0,
            regions,
            staging: Vec::new(),
            encode: Box::new(move |encoder, regions, _staging| encode(encoder, regions)),
        })
    }

    pub(crate) fn with_work_shape(
        mut self,
        batching_form: DeviceBatchingForm,
        participant_count: u32,
        token_count: u64,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        if participant_count == 0 {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal command attribution has zero participants",
            ));
        }
        self.batching_form = batching_form;
        self.participant_count = participant_count;
        self.token_count = token_count;
        Ok(self)
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
            batching_form: DeviceBatchingForm::Scalar,
            participant_count: 0,
            token_count: 0,
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
    CommandBuffer,
}

pub struct MetalDeviceFence {
    submission_id: u64,
    command_buffer: CommandBuffer,
    timing: MetalFenceTiming,
    command_timing: MetalFenceCommandTiming,
    stream_state: Arc<MetalStreamState>,
    pending: Arc<MetalPendingSubmissions>,
    terminal_accounted: AtomicBool,
    commands: Vec<MetalDeviceCommand>,
    attribution: Option<DeviceSubmissionAttribution>,
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

    fn execution_timing(&self) -> DeviceTimingMeasurement<DeviceExecutionTiming> {
        match self.timing {
            MetalFenceTiming::NotRequested => DeviceTimingMeasurement::NotRequested,
            MetalFenceTiming::CommandBuffer => {
                metal_command_buffer_execution_timing(&self.command_buffer)
            }
        }
    }

    fn terminal_receipt<E>(&self, terminal: DeviceTerminal<E>) -> DeviceTerminalReceipt<E> {
        match self.timing {
            MetalFenceTiming::NotRequested => DeviceTerminalReceipt::unprofiled(terminal),
            MetalFenceTiming::CommandBuffer => {
                DeviceTerminalReceipt::profiled_with_submission_timing(
                    terminal,
                    self.execution_timing(),
                    self.command_timing.measurement(&self.command_buffer),
                )
            }
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

fn metal_command_buffer_execution_timing(
    command_buffer: &CommandBufferRef,
) -> DeviceTimingMeasurement<DeviceExecutionTiming> {
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return DeviceTimingMeasurement::Unavailable(
            DeviceTimingUnavailableReason::BackendMeasurementFailed,
        );
    }
    let start_selector = sel!(GPUStartTime);
    let end_selector = sel!(GPUEndTime);
    let (supports_start, supports_end): (BOOL, BOOL) = unsafe {
        (
            msg_send![command_buffer, respondsToSelector: start_selector],
            msg_send![command_buffer, respondsToSelector: end_selector],
        )
    };
    if supports_start != YES || supports_end != YES {
        return DeviceTimingMeasurement::Unavailable(
            DeviceTimingUnavailableReason::BackendUnsupported,
        );
    }
    let (started_seconds, ended_seconds): (f64, f64) = unsafe {
        (
            msg_send![command_buffer, GPUStartTime],
            msg_send![command_buffer, GPUEndTime],
        )
    };
    if !started_seconds.is_finite() || !ended_seconds.is_finite() || ended_seconds < started_seconds
    {
        return DeviceTimingMeasurement::Unavailable(
            DeviceTimingUnavailableReason::BackendMeasurementFailed,
        );
    }
    let elapsed_ns = (ended_seconds - started_seconds) * 1_000_000_000.0;
    if !elapsed_ns.is_finite() || elapsed_ns > u64::MAX as f64 {
        return DeviceTimingMeasurement::Unavailable(
            DeviceTimingUnavailableReason::DurationOverflow,
        );
    }
    DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
        elapsed_ns.round() as u64,
    ))
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
    timestamp_counter_support: OnceLock<MetalTimestampCounterSupport>,
    static_weight_import_gate: Mutex<()>,
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
            timestamp_counter_support: OnceLock::new(),
            static_weight_import_gate: Mutex::new(()),
        })
    }

    pub(crate) fn device(&self) -> &metal::Device {
        &self.device
    }

    fn timestamp_counter_support(&self) -> MetalTimestampCounterSupport {
        self.timestamp_counter_support
            .get_or_init(|| timestamp_counter_support(&self.device))
            .clone()
    }

    fn allocate_request(
        &self,
        request: &BufferRequest,
    ) -> Result<MetalDeviceBuffer, MetalDeviceRuntimeError> {
        let descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        if request.usage() == BufferUsage::Weights {
            return Ok(MetalDeviceBuffer {
                descriptor,
                runtime_instance: self.runtime_instance,
                backing: MetalDeviceBufferBacking::StaticWeights(Arc::new(
                    MetalStaticWeightArena {
                        sealed: OnceLock::new(),
                    },
                )),
            });
        }
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
            descriptor,
            runtime_instance: self.runtime_instance,
            backing: MetalDeviceBufferBacking::Contiguous(Arc::new(MetalAllocation {
                base,
                aligned_offset_bytes,
                requested_bytes: request.size_bytes(),
                _retained_host_memory: None,
            })),
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
        entries: Vec<(DeviceCommandPhase, Option<u32>, MetalDeviceCommand)>,
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
        if entries.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                MetalDeviceRuntimeError::contract("Metal command batch is empty"),
            ));
        }
        if u32::try_from(entries.len()).is_err() {
            return Err(DefinitelyNotSubmitted::new(
                MetalDeviceRuntimeError::contract("Metal command batch exceeds u32 indexing"),
            ));
        }
        if entries
            .iter()
            .any(|(_, _, command)| command.runtime_instance != self.runtime_instance)
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
            DeviceTimingMode::Completion | DeviceTimingMode::Replay | DeviceTimingMode::Kernel => {
                MetalFenceTiming::CommandBuffer
            }
        };
        drop(begin_timing_stage);

        let command_buffer = stream.queue.new_command_buffer().to_owned();
        let physical_span_attribution = timing_mode.physical_span_attribution_enabled();
        let kernel_attribution = timing_mode.kernel_attribution_enabled();
        if kernel_attribution {
            command_buffer.set_label("ferrum.vnext.full_profile");
        } else if physical_span_attribution {
            command_buffer.set_label("ferrum.vnext.replay_profile");
        }
        let counter_capture = if physical_span_attribution {
            match self.timestamp_counter_support() {
                MetalTimestampCounterSupport::Supported(counter_set) => Some(
                    MetalCounterCaptureBuilder::new(self.device.clone(), counter_set),
                ),
                MetalTimestampCounterSupport::Unsupported => None,
            }
        } else {
            None
        };
        let mut encoder = MetalSubmissionEncoder::new(
            &command_buffer,
            physical_span_attribution,
            counter_capture,
        );
        let mut attribution = kernel_attribution.then(|| Vec::with_capacity(entries.len()));
        let enqueue_stage =
            MetalSubmissionStageTimer::start(timing_sink, DeviceSubmissionStage::EnqueueCommands);
        for (command_index, (phase, node_index, command)) in entries.iter().enumerate() {
            let command_index = command_index as u32;
            encoder.begin_command(command_index, command.operation);
            if let Err(error) = command.encode(&mut encoder) {
                stream.state.cancel_recording();
                return Err(DefinitelyNotSubmitted::new(error));
            }
            if let Some(attribution) = attribution.as_mut() {
                let (compute_dispatch_count, transfer_command_count) = encoder.command_counts();
                if let Some(observation) = DeviceNativeWorkAttribution::new(
                    command_index,
                    *node_index,
                    *phase,
                    command.operation,
                    DeviceExecutionPath::Eager,
                    command.batching_form,
                    command.participant_count,
                    command.token_count,
                    compute_dispatch_count,
                    transfer_command_count,
                    None,
                ) {
                    attribution.push(observation);
                }
            }
        }
        let counter_capture = encoder.finish();
        let command_timing = match (physical_span_attribution, counter_capture) {
            (false, _) => MetalFenceCommandTiming::NotRequested,
            (true, Some(capture)) => MetalFenceCommandTiming::Captured {
                capture,
                resolved: OnceLock::new(),
            },
            (true, None) => MetalFenceCommandTiming::Unavailable(
                DeviceTimingUnavailableReason::BackendUnsupported,
            ),
        };
        drop(enqueue_stage);
        let attribution = attribution.and_then(DeviceSubmissionAttribution::new);
        let commands = entries.into_iter().map(|(_, _, command)| command).collect();

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
            command_timing,
            stream_state: Arc::clone(&stream.state),
            pending: Arc::clone(&stream.pending),
            terminal_accounted: AtomicBool::new(false),
            commands,
            attribution,
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

    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        let permit = match self.static_weight_import_gate.try_lock() {
            Ok(permit) => permit,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Some(Err(MetalDeviceRuntimeError::contract(
                    "another Metal static-weight import transaction is active",
                )))
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Some(Err(MetalDeviceRuntimeError::contract(
                    "Metal static-weight import gate is poisoned",
                )))
            }
        };
        Some(
            MetalStaticWeightImport::new(self.device.clone(), self.runtime_instance, permit)
                .map(|import| Box::new(import) as Box<_>),
        )
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
        if destination.descriptor.usage == BufferUsage::Weights {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static weights are immutable after import",
            ));
        }
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
        if destination.descriptor.usage == BufferUsage::Weights {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static weights must use the import transaction",
            ));
        }
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
        if destination.descriptor.usage == BufferUsage::Weights {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal static weights must use the import transaction",
            ));
        }
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
        let entries = commands
            .into_entries()
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .collect();
        self.submit_commands(stream, entries, timing_mode, timing_sink)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        fence.attribution.clone()
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
        BufferUsage, DynamicStorageAllocator, DynamicStorageView, ResourceId, StableHostMemory,
        WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightId,
    };
    use std::ptr::NonNull;

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

    fn weight_buffer_request(resource: &str, size_bytes: u64) -> BufferRequest {
        BufferRequest::new(
            ResourceId::new(resource).expect("resource id"),
            size_bytes,
            64,
            BufferUsage::Weights,
            ElementType::U8,
        )
        .expect("weight buffer request")
    }

    fn weight_component(id: &str, external_name: &str, size_bytes: usize) -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new(id).expect("weight id"),
            role: WeightComponentRole::Values,
            external_names: vec![external_name.to_owned()],
            dimensions: vec![size_bytes as u64],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
            required: true,
        }
    }

    fn weight_payload<'a>(
        component: &WeightComponentSpec,
        bytes: &'a [u8],
    ) -> WeightComponentPayload<'a> {
        WeightComponentPayload::new(
            component,
            component.external_names[0].clone(),
            "model.gguf",
            component.dimensions.clone(),
            ElementType::U8,
            bytes,
        )
        .expect("weight payload")
    }

    fn region_bytes(region: &MetalBufferRegion) -> &[u8] {
        let start = usize::try_from(region.offset_bytes()).expect("region offset");
        let length = usize::try_from(region.length_bytes()).expect("region length");
        // SAFETY: MetalBufferRegion retains the allocation and validates that
        // this exact byte range is within it.
        unsafe { std::slice::from_raw_parts(region.buffer().contents().add(start).cast(), length) }
    }

    struct PageMemory {
        pointer: NonNull<u8>,
        length: usize,
    }

    impl PageMemory {
        fn new(length: usize) -> Self {
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(pointer, libc::MAP_FAILED);
            let pointer = NonNull::new(pointer.cast::<u8>()).expect("mmap pointer");
            for index in 0..length {
                unsafe { pointer.as_ptr().add(index).write((index % 251) as u8) };
            }
            Self { pointer, length }
        }
    }

    impl Drop for PageMemory {
        fn drop(&mut self) {
            let result = unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.length) };
            assert_eq!(result, 0);
        }
    }

    unsafe impl Send for PageMemory {}
    unsafe impl Sync for PageMemory {}

    // SAFETY: the mmap is fixed-size and is never mutated after construction.
    unsafe impl StableHostMemory for PageMemory {
        fn stable_bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
        }
    }

    fn compute_entries(
        commands: Vec<MetalDeviceCommand>,
    ) -> Vec<(DeviceCommandPhase, Option<u32>, MetalDeviceCommand)> {
        commands
            .into_iter()
            .map(|command| (DeviceCommandPhase::Compute, None, command))
            .collect()
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
        let source_allocation = source.contiguous_allocation();
        let source_address = source_allocation.base.contents() as usize
            + usize::try_from(source_allocation.aligned_offset_bytes).expect("aligned offset");
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
                compute_entries(commands),
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("submission");
        assert!(runtime.submission_attribution(&fence).is_none());
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
    fn static_weight_import_publishes_sorted_component_regions_atomically() {
        let runtime = runtime();
        let weights = runtime
            .allocate_request(&weight_buffer_request("resource/weights", 80))
            .expect("logical weight arena");
        let left = weight_component("component.left", "left.weight", 8);
        let right = weight_component("component.right", "right.weight", 8);
        let left_bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let right_bytes = [11_u8, 12, 13, 14, 15, 16, 17, 18];
        let left_payload = weight_payload(&left, &left_bytes);
        let right_payload = weight_payload(&right, &right_bytes);
        let mut import = runtime
            .begin_static_weight_import()
            .expect("Metal import support")
            .expect("Metal import session");
        assert!(
            runtime
                .begin_static_weight_import()
                .expect("Metal import support")
                .is_err(),
            "one runtime must not admit concurrent import transactions"
        );
        import
            .import_component(&right_payload, &weights, 64)
            .expect("right component");
        import
            .import_component(&left_payload, &weights, 0)
            .expect("left component");
        assert!(
            weights.region(0..8).is_err(),
            "unsealed arena must be hidden"
        );
        import.seal().expect("seal weight import");
        assert!(
            runtime
                .begin_static_weight_import()
                .expect("Metal import support")
                .is_ok(),
            "sealing must release the runtime import permit"
        );

        assert_eq!(region_bytes(&weights.region(0..8).unwrap()), left_bytes);
        assert_eq!(region_bytes(&weights.region(64..72).unwrap()), right_bytes);
        assert!(weights.region(8..64).is_err(), "padding is not a tensor");
        assert!(
            weights.region(4..68).is_err(),
            "one binding cannot cross tensors"
        );
        assert!(runtime
            .encode_upload(
                &left_bytes,
                HostTransferLayout::new(ElementType::U8, 8).unwrap(),
                &weights,
                0,
            )
            .is_err());
    }

    #[test]
    fn failed_static_weight_import_does_not_publish_and_can_be_retried() {
        let runtime = runtime();
        let weights = runtime
            .allocate_request(&weight_buffer_request("resource/retry-weights", 16))
            .expect("logical weight arena");
        let component = weight_component("component.retry", "retry.weight", 8);
        let bytes = [3_u8; 8];
        let payload = weight_payload(&component, &bytes);
        let mut overlapping = runtime
            .begin_static_weight_import()
            .unwrap()
            .expect("first import session");
        overlapping.import_component(&payload, &weights, 0).unwrap();
        overlapping.import_component(&payload, &weights, 0).unwrap();
        assert!(overlapping.seal().is_err());
        assert!(weights.region(0..8).is_err());

        let mut retry = runtime
            .begin_static_weight_import()
            .unwrap()
            .expect("retry import session");
        assert!(retry.import_component(&payload, &weights, 9).is_err());
        retry.import_component(&payload, &weights, 0).unwrap();
        retry.seal().expect("retry seal");
        assert_eq!(region_bytes(&weights.region(0..8).unwrap()), bytes);
    }

    #[test]
    fn static_weight_no_copy_region_retains_page_owner_after_source_drop() {
        let runtime = runtime();
        let weights = runtime
            .allocate_request(&weight_buffer_request("resource/mmap-weights", 8))
            .expect("logical weight arena");
        let page_size = system_page_size().expect("page size");
        let owner = Arc::new(PageMemory::new(page_size));
        let retained = RetainedHostMemoryRegion::new(Arc::clone(&owner), 128, 8).unwrap();
        let component = weight_component("component.mmap", "mmap.weight", 8);
        let payload = weight_payload(&component, retained.bytes())
            .with_retained_host_memory(retained.clone())
            .unwrap();
        let expected = payload.bytes().to_vec();
        let mut import = runtime
            .begin_static_weight_import()
            .unwrap()
            .expect("Metal import session");
        import.import_component(&payload, &weights, 0).unwrap();
        import.seal().unwrap();
        let region = weights.region(0..8).unwrap();
        assert!(region.allocation._retained_host_memory.is_some());
        drop(payload);
        drop(retained);
        drop(owner);
        assert_eq!(region_bytes(&region), expected);
    }

    #[test]
    fn static_weight_import_copies_retained_regions_without_a_safe_native_view() {
        let runtime = runtime();
        let page_size = system_page_size().expect("page size");
        for (suffix, owner_length, region_offset) in [
            ("misaligned", page_size, 1_usize),
            ("partial-tail-page", page_size + 8, page_size),
        ] {
            let weights = runtime
                .allocate_request(&weight_buffer_request(&format!("resource/{suffix}"), 8))
                .expect("logical weight arena");
            let owner = Arc::new(PageMemory::new(owner_length));
            let retained =
                RetainedHostMemoryRegion::new(Arc::clone(&owner), region_offset, 8).unwrap();
            let component = weight_component(
                &format!("component.{suffix}"),
                &format!("{suffix}.weight"),
                8,
            );
            let payload = weight_payload(&component, retained.bytes())
                .with_retained_host_memory(retained.clone())
                .unwrap();
            let expected = payload.bytes().to_vec();
            let mut import = runtime
                .begin_static_weight_import()
                .unwrap()
                .expect("Metal import session");
            import.import_component(&payload, &weights, 0).unwrap();
            import.seal().unwrap();
            let region = weights.region(0..8).unwrap();
            assert!(region.allocation._retained_host_memory.is_none());
            let address = region.buffer().contents() as usize + region.offset_bytes() as usize;
            assert_eq!(address % 64, 0);
            drop(payload);
            drop(owner);
            assert_eq!(region_bytes(&region), expected);
        }
    }

    #[test]
    fn profiled_transfer_reports_metal_command_buffer_time() {
        let runtime = runtime();
        let destination = runtime
            .allocate_request(&buffer_request("resource/profiled-destination"))
            .expect("destination allocation");
        let mut stream = runtime.create_stream().expect("stream");
        let command = runtime
            .encode_zero(&destination, 0, 8)
            .expect("zero command");

        let fence = runtime
            .submit_commands(
                &mut stream,
                compute_entries(vec![command]),
                DeviceTimingMode::Completion,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("profiled submission");
        let terminal = runtime.wait_fence(&fence).expect("terminal fence");

        assert!(terminal.terminal().is_succeeded());
        assert!(matches!(
            terminal.execution_timing(),
            DeviceTimingMeasurement::Measured(timing) if timing.elapsed_ns() > 0
        ));
    }

    #[test]
    fn kernel_profile_attributes_node_bound_native_work() {
        let runtime = runtime();
        let destination = runtime
            .allocate_request(&buffer_request("resource/kernel-profile-destination"))
            .expect("destination allocation");
        let command = runtime
            .encode_zero(&destination, 0, 8)
            .expect("zero command")
            .with_work_shape(DeviceBatchingForm::Packed, 2, 8)
            .expect("work shape");
        let mut stream = runtime.create_stream().expect("stream");

        let fence = runtime
            .submit_commands(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                DeviceTimingMode::Kernel,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("kernel-profiled submission");
        let attribution = runtime
            .submission_attribution(&fence)
            .expect("kernel profile must retain native work attribution");
        let [command] = attribution.commands() else {
            panic!("expected exactly one native command attribution")
        };
        assert_eq!(command.command_index(), 0);
        assert_eq!(command.node_index(), Some(0));
        assert_eq!(command.command_phase(), DeviceCommandPhase::Compute);
        assert_eq!(command.native_op_id(), "device zero");
        assert_eq!(command.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(command.batching_form(), DeviceBatchingForm::Packed);
        assert_eq!(command.participant_count(), 2);
        assert_eq!(command.token_count(), 8);
        assert_eq!(command.compute_dispatch_count(), 0);
        assert_eq!(command.transfer_command_count(), 1);

        let terminal = runtime.wait_fence(&fence).expect("terminal fence");
        assert!(terminal.terminal().is_succeeded());
        assert!(matches!(
            terminal.execution_timing(),
            DeviceTimingMeasurement::Measured(timing) if timing.elapsed_ns() > 0
        ));
        let DeviceTimingMeasurement::Measured(timing) = terminal.submission_timing() else {
            panic!("Metal kernel profile must report command counter timing")
        };
        let [timing] = timing.spans() else {
            panic!("expected exactly one command timing row")
        };
        assert_eq!(timing.start_command_index(), command.command_index());
        assert_eq!(timing.end_command_index(), command.command_index() + 1);
        let intervals = timing.measurement().intervals().unwrap();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].kind(), DeviceExecutionIntervalKind::Transfer);
        assert!(timing.measurement().elapsed_ns().unwrap() > 0);
    }

    #[test]
    fn replay_profile_reports_physical_timing_without_native_attribution() {
        let runtime = runtime();
        let destination = runtime
            .allocate_request(&buffer_request("resource/replay-profile-destination"))
            .expect("destination allocation");
        let command = runtime
            .encode_zero(&destination, 0, 8)
            .expect("zero command");
        let mut stream = runtime.create_stream().expect("stream");

        let fence = runtime
            .submit_commands(
                &mut stream,
                compute_entries(vec![command]),
                DeviceTimingMode::Replay,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("replay-profiled submission");
        assert!(runtime.submission_attribution(&fence).is_none());

        let terminal = runtime.wait_fence(&fence).expect("terminal fence");
        assert!(terminal.terminal().is_succeeded());
        assert!(matches!(
            terminal.execution_timing(),
            DeviceTimingMeasurement::Measured(timing) if timing.elapsed_ns() > 0
        ));
        let DeviceTimingMeasurement::Measured(timing) = terminal.submission_timing() else {
            panic!("Metal replay profile must report physical command timing")
        };
        assert_eq!(timing.command_count(), 1);
        assert_eq!(timing.spans().len(), 1);
    }

    #[test]
    fn kernel_profile_counts_each_command_inside_one_blit_encoder() {
        let runtime = runtime();
        let destination = runtime
            .allocate_request(&buffer_request("resource/kernel-profile-batched-transfer"))
            .expect("destination allocation");
        let region = destination.region(0..8).expect("destination region");
        let command = MetalDeviceCommand::operation(
            "batched device zero",
            vec![region],
            |encoder, regions| {
                encoder.with_blit_commands(3, |blit| {
                    for _ in 0..3 {
                        blit.fill_buffer(
                            regions[0].buffer(),
                            NSRange::new(regions[0].offset_bytes(), regions[0].length_bytes()),
                            0,
                        );
                    }
                });
                Ok(())
            },
        )
        .expect("batched transfer command")
        .with_work_shape(DeviceBatchingForm::Packed, 3, 3)
        .expect("work shape");
        let mut stream = runtime.create_stream().expect("stream");

        let fence = runtime
            .submit_commands(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                DeviceTimingMode::Kernel,
                &DisabledDeviceSubmissionTimingSink,
            )
            .expect("kernel-profiled submission");
        let attribution = runtime
            .submission_attribution(&fence)
            .expect("kernel profile must retain native work attribution");
        let [command] = attribution.commands() else {
            panic!("expected exactly one native command attribution")
        };
        assert_eq!(command.compute_dispatch_count(), 0);
        assert_eq!(command.transfer_command_count(), 3);

        let terminal = runtime.wait_fence(&fence).expect("terminal fence");
        assert!(terminal.terminal().is_succeeded());
        let DeviceTimingMeasurement::Measured(timing) = terminal.submission_timing() else {
            panic!("Metal kernel profile must report command counter timing")
        };
        let [timing] = timing.spans() else {
            panic!("expected exactly one command timing row")
        };
        assert_eq!(timing.start_command_index(), command.command_index());
        assert_eq!(timing.end_command_index(), command.command_index() + 1);
        let intervals = timing.measurement().intervals().unwrap();
        assert_eq!(
            intervals.len(),
            1,
            "three transfer operations in one blit encoder are one physical interval"
        );
        assert_eq!(intervals[0].kind(), DeviceExecutionIntervalKind::Transfer);
        assert!(timing.measurement().elapsed_ns().unwrap() > 0);
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
                compute_entries(vec![command]),
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
