use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrum_interfaces::vnext::{
    BufferDescriptor, CapabilityId, CopyRegion, DefinitelyNotSubmitted, DeviceAllocationPermit,
    DeviceBatchingForm, DeviceBufferRetention, DeviceClass, DeviceCommandBatch,
    DeviceCommandLogicalWork, DeviceCommandPhase, DeviceComputePathRequirement, DeviceDescriptor,
    DeviceErrorReport, DeviceExecutionPath, DeviceNativeOperationId, DeviceNativeWorkAttribution,
    DeviceRuntime, DeviceSubmissionAttribution, DeviceTerminal, DeviceTerminalReceipt,
    DeviceTimingMode, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout, StreamState,
    VNextError, DENSE_LINEAR_F16_CAPABILITY_ID, DEVICE_COPY_NATIVE_OPERATION_ID,
    DEVICE_ZERO_NATIVE_OPERATION_ID, HOST_UPLOAD_NATIVE_OPERATION_ID,
};
use half::f16;

static NEXT_RUNTIME_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_STREAM_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_FENCE_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ReferenceDeviceRuntimeConfig {
    pub(crate) descriptor: DeviceDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceDeviceRuntimeError {
    message: String,
}

impl ReferenceDeviceRuntimeError {
    pub(crate) fn contract(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ReferenceDeviceRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ReferenceDeviceRuntimeError {}

struct ReferenceAllocation {
    bytes: Mutex<Box<[u8]>>,
    logical_offset_bytes: usize,
    live_allocations: Arc<AtomicU64>,
}

impl ReferenceAllocation {
    fn lock(&self) -> MutexGuard<'_, Box<[u8]>> {
        self.bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for ReferenceAllocation {
    fn drop(&mut self) {
        let prior = self.live_allocations.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prior > 0, "reference allocation accounting underflow");
    }
}

/// One core-authorized reference allocation.
pub struct ReferenceDeviceBuffer {
    descriptor: BufferDescriptor,
    runtime_instance: u64,
    allocation: Arc<ReferenceAllocation>,
}

impl fmt::Debug for ReferenceDeviceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceBuffer")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl ReferenceDeviceBuffer {
    fn region(
        &self,
        range: Range<u64>,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        self.region_with_retention(range, None)
    }

    pub(crate) fn retained_region(
        &self,
        range: Range<u64>,
        retention: DeviceBufferRetention,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        self.region_with_retention(range, Some(retention))
    }

    fn region_with_retention(
        &self,
        range: Range<u64>,
        retention: Option<DeviceBufferRetention>,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        if range.start >= range.end || range.end > self.descriptor.size_bytes {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference buffer region is empty or exceeds its admitted allocation",
            ));
        }
        let offset_bytes = usize::try_from(range.start).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference buffer offset exceeds usize")
        })?;
        let offset_bytes = self
            .allocation
            .logical_offset_bytes
            .checked_add(offset_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference buffer physical offset overflows usize",
                )
            })?;
        let length_bytes = usize::try_from(range.end - range.start).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference buffer length exceeds usize")
        })?;
        Ok(ReferenceBufferRegion {
            allocation: Arc::clone(&self.allocation),
            runtime_instance: self.runtime_instance,
            offset_bytes,
            length_bytes,
            element_type: self.descriptor.element_type,
            _retention: retention,
        })
    }
}

#[derive(Clone)]
pub(crate) struct ReferenceBufferRegion {
    allocation: Arc<ReferenceAllocation>,
    runtime_instance: u64,
    offset_bytes: usize,
    length_bytes: usize,
    element_type: ElementType,
    _retention: Option<DeviceBufferRetention>,
}

impl ReferenceBufferRegion {
    pub(crate) const fn length_bytes(&self) -> usize {
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

    fn validate_runtime(&self, runtime_instance: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.runtime_instance != runtime_instance {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference command contains a buffer from another runtime",
            ));
        }
        Ok(())
    }

    fn read(&self) -> Vec<u8> {
        let bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].to_vec()
    }

    fn write(&self, source: &[u8]) {
        assert_eq!(source.len(), self.length_bytes);
        let mut bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].copy_from_slice(source);
    }

    fn zero(&self) {
        let mut bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].fill(0);
    }
}

pub(crate) struct ReferenceDenseLinearLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) weight: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) rows: usize,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

enum ReferenceCommandKind {
    Copy {
        source: ReferenceBufferRegion,
        destination: ReferenceBufferRegion,
    },
    Upload {
        source: Box<[u8]>,
        destination: ReferenceBufferRegion,
    },
    Zero {
        destination: ReferenceBufferRegion,
    },
    DenseLinear {
        launches: Box<[ReferenceDenseLinearLaunch]>,
    },
}

/// An owned, fully validated command for the synchronous reference runtime.
pub struct ReferenceDeviceCommand {
    operation: &'static str,
    batching_form: DeviceBatchingForm,
    participant_start: u32,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    kind: ReferenceCommandKind,
}

impl fmt::Debug for ReferenceDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ReferenceCommandKind::Copy { .. } => "copy",
            ReferenceCommandKind::Upload { .. } => "upload",
            ReferenceCommandKind::Zero { .. } => "zero",
            ReferenceCommandKind::DenseLinear { .. } => "dense_linear",
        };
        formatter
            .debug_struct("ReferenceDeviceCommand")
            .field("kind", &kind)
            .field("operation", &self.operation)
            .field("batching_form", &self.batching_form)
            .field("participant_start", &self.participant_start)
            .field("participant_count", &self.participant_count)
            .field("token_count", &self.token_count)
            .finish()
    }
}

impl ReferenceDeviceCommand {
    pub(crate) fn dense_linear(
        launches: Vec<ReferenceDenseLinearLaunch>,
        batching_form: DeviceBatchingForm,
        participant_count: u32,
        token_count: u64,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        if launches.is_empty() || participant_count == 0 || token_count == 0 {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference dense-linear attribution has no participants or work",
            ));
        }
        let compute_dispatch_count = u64::try_from(launches.len()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract(
                "reference dense-linear dispatch count exceeds u64",
            )
        })?;
        Ok(Self {
            operation: "vnext_dense_linear",
            batching_form,
            participant_start: 0,
            participant_count,
            token_count,
            compute_dispatch_count,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::DenseLinear {
                launches: launches.into_boxed_slice(),
            },
        })
    }

    fn transfer(operation: &'static str, kind: ReferenceCommandKind) -> Self {
        Self {
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            kind,
        }
    }

    fn bind_core_logical_work(
        mut self,
        logical_work: DeviceCommandLogicalWork,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        if self.participant_count != 0 || self.token_count != 0 {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference core logical work cannot replace provider command attribution",
            ));
        }
        self.batching_form = logical_work.batching_form();
        self.participant_start = logical_work.participant_start();
        self.participant_count = logical_work.participant_count();
        self.token_count = logical_work.token_count();
        Ok(self)
    }

    fn attribution(
        &self,
        command_index: u32,
        node_index: Option<u32>,
        phase: DeviceCommandPhase,
    ) -> Result<DeviceNativeWorkAttribution, ReferenceDeviceRuntimeError> {
        let native_op_id = DeviceNativeOperationId::new(self.operation).ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract(
                "reference command attribution has a non-portable native operation identity",
            )
        })?;
        DeviceNativeWorkAttribution::with_participant_range(
            command_index,
            node_index,
            phase,
            native_op_id,
            DeviceExecutionPath::Eager,
            self.batching_form,
            self.participant_start,
            self.participant_count,
            self.token_count,
            self.compute_dispatch_count,
            self.transfer_command_count,
            None,
        )
        .ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract(
                "reference command attribution has invalid native work metadata",
            )
        })
    }

    fn validate_runtime(&self, runtime_instance: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        match &self.kind {
            ReferenceCommandKind::Copy {
                source,
                destination,
            } => {
                source.validate_runtime(runtime_instance)?;
                destination.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::Upload { destination, .. }
            | ReferenceCommandKind::Zero { destination } => {
                destination.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::DenseLinear { launches } => {
                for launch in launches {
                    launch.input.validate_runtime(runtime_instance)?;
                    launch.weight.validate_runtime(runtime_instance)?;
                    launch.output.validate_runtime(runtime_instance)?;
                }
                Ok(())
            }
        }
    }

    fn execute(&self, counters: &ReferenceRuntimeCounters) {
        match &self.kind {
            ReferenceCommandKind::Copy {
                source,
                destination,
            } => destination.write(&source.read()),
            ReferenceCommandKind::Upload {
                source,
                destination,
            } => {
                destination.write(source);
                counters.uploaded_bytes.fetch_add(
                    u64::try_from(source.len()).expect("reference upload length fits u64"),
                    Ordering::Relaxed,
                );
            }
            ReferenceCommandKind::Zero { destination } => destination.zero(),
            ReferenceCommandKind::DenseLinear { launches } => {
                for launch in launches {
                    execute_dense_linear(launch);
                    counters
                        .dense_linear_launches
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

fn execute_dense_linear(launch: &ReferenceDenseLinearLaunch) {
    let input = launch.input.read();
    let weight = launch.weight.read();
    let mut output = vec![0_u8; launch.output.length_bytes()];
    for row in 0..launch.rows {
        for out_feature in 0..launch.out_features {
            let mut sum = 0.0_f32;
            for in_feature in 0..launch.in_features {
                let input_index = row * launch.in_features + in_feature;
                let weight_index = out_feature * launch.in_features + in_feature;
                sum += read_f16(&input, input_index) * read_f16(&weight, weight_index);
            }
            let output_index = row * launch.out_features + out_feature;
            let bytes = f16::from_f32(sum).to_bits().to_le_bytes();
            output[output_index * 2..output_index * 2 + 2].copy_from_slice(&bytes);
        }
    }
    launch.output.write(&output);
}

fn read_f16(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn aligned_storage(
    size_bytes: usize,
    alignment_bytes: usize,
) -> Result<(Box<[u8]>, usize), ReferenceDeviceRuntimeError> {
    if size_bytes == 0 || alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference allocation size or alignment is invalid",
        ));
    }
    let storage_bytes = size_bytes.checked_add(alignment_bytes - 1).ok_or_else(|| {
        ReferenceDeviceRuntimeError::contract("reference aligned allocation size overflows")
    })?;
    let bytes = vec![0_u8; storage_bytes].into_boxed_slice();
    let base = bytes.as_ptr() as usize;
    let aligned = base
        .checked_add(alignment_bytes - 1)
        .map(|address| address & !(alignment_bytes - 1))
        .ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract("reference aligned address overflows")
        })?;
    let logical_offset_bytes = aligned.checked_sub(base).ok_or_else(|| {
        ReferenceDeviceRuntimeError::contract("reference aligned address precedes its allocation")
    })?;
    if logical_offset_bytes
        .checked_add(size_bytes)
        .is_none_or(|end| end > bytes.len())
    {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference aligned logical range exceeds its allocation",
        ));
    }
    Ok((bytes, logical_offset_bytes))
}

pub struct ReferenceDeviceStream {
    runtime_instance: u64,
    stream_instance: u64,
    state: StreamState,
}

impl fmt::Debug for ReferenceDeviceStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceStream")
            .field("runtime_instance", &self.runtime_instance)
            .field("stream_instance", &self.stream_instance)
            .field("state", &self.state)
            .finish()
    }
}

pub struct ReferenceDeviceFence {
    runtime_instance: u64,
    fence_instance: u64,
    attribution: Option<DeviceSubmissionAttribution>,
}

impl fmt::Debug for ReferenceDeviceFence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceFence")
            .field("runtime_instance", &self.runtime_instance)
            .field("fence_instance", &self.fence_instance)
            .finish()
    }
}

#[derive(Default)]
struct ReferenceRuntimeCounters {
    allocations: AtomicU64,
    live_allocations: Arc<AtomicU64>,
    submissions: AtomicU64,
    commands: AtomicU64,
    uploaded_bytes: AtomicU64,
    dense_linear_launches: AtomicU64,
    readback_bytes: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDeviceRuntimeSnapshot {
    pub allocations: u64,
    pub live_allocations: u64,
    pub submissions: u64,
    pub commands: u64,
    pub uploaded_bytes: u64,
    pub dense_linear_launches: u64,
    pub readback_bytes: u64,
}

/// Synchronous, in-memory vNext runtime used for bounded numerical reference
/// execution. It is never selected implicitly by a product backend.
pub struct ReferenceDeviceRuntime {
    descriptor: DeviceDescriptor,
    runtime_instance: u64,
    counters: ReferenceRuntimeCounters,
}

impl ReferenceDeviceRuntime {
    pub(crate) fn new(
        config: ReferenceDeviceRuntimeConfig,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        config
            .descriptor
            .validate()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let supported_capabilities =
            BTreeSet::from([CapabilityId::new(DENSE_LINEAR_F16_CAPABILITY_ID)
                .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?]);
        if config.descriptor.class != DeviceClass::Reference
            || config.descriptor.capabilities != supported_capabilities
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime descriptor overclaims its fixed device class or capabilities",
            ));
        }
        let runtime_instance = NEXT_RUNTIME_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ReferenceDeviceRuntimeError::contract("reference runtime identity exhausted")
            })?;
        Ok(Self {
            descriptor: config.descriptor,
            runtime_instance,
            counters: ReferenceRuntimeCounters::default(),
        })
    }

    pub fn snapshot(&self) -> ReferenceDeviceRuntimeSnapshot {
        ReferenceDeviceRuntimeSnapshot {
            allocations: self.counters.allocations.load(Ordering::Relaxed),
            live_allocations: self.counters.live_allocations.load(Ordering::Relaxed),
            submissions: self.counters.submissions.load(Ordering::Relaxed),
            commands: self.counters.commands.load(Ordering::Relaxed),
            uploaded_bytes: self.counters.uploaded_bytes.load(Ordering::Relaxed),
            dense_linear_launches: self.counters.dense_linear_launches.load(Ordering::Relaxed),
            readback_bytes: self.counters.readback_bytes.load(Ordering::Relaxed),
        }
    }
}

impl DeviceRuntime for ReferenceDeviceRuntime {
    type Buffer = ReferenceDeviceBuffer;
    type Stream = ReferenceDeviceStream;
    type Command = ReferenceDeviceCommand;
    type Fence = ReferenceDeviceFence;
    type Error = ReferenceDeviceRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        let size = usize::try_from(request.size_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference allocation exceeds usize")
        })?;
        let alignment = usize::try_from(request.alignment_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference alignment exceeds usize")
        })?;
        let (bytes, logical_offset_bytes) = aligned_storage(size, alignment)?;
        let descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        self.counters.allocations.fetch_add(1, Ordering::Relaxed);
        self.counters
            .live_allocations
            .fetch_add(1, Ordering::Relaxed);
        Ok(ReferenceDeviceBuffer {
            descriptor,
            runtime_instance: self.runtime_instance,
            allocation: Arc::new(ReferenceAllocation {
                bytes: Mutex::new(bytes),
                logical_offset_bytes,
                live_allocations: Arc::clone(&self.counters.live_allocations),
            }),
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        let stream_instance = NEXT_STREAM_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ReferenceDeviceRuntimeError::contract("reference stream identity exhausted")
            })?;
        Ok(ReferenceDeviceStream {
            runtime_instance: self.runtime_instance,
            stream_instance,
            state: StreamState::Ready,
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.runtime_instance == self.runtime_instance {
            stream.state
        } else {
            StreamState::Failed
        }
    }

    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        region
            .validate_bounds(&source.descriptor, &destination.descriptor)
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if source.descriptor.element_type != destination.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference copy requires matching source and destination element types",
            ));
        }
        let source = source.region(
            region.source_offset_bytes()..region.source_offset_bytes() + region.length_bytes(),
        )?;
        let destination = destination.region(
            region.destination_offset_bytes()
                ..region.destination_offset_bytes() + region.length_bytes(),
        )?;
        Ok(ReferenceDeviceCommand::transfer(
            DEVICE_COPY_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Copy {
                source,
                destination,
            },
        ))
    }

    fn encode_upload(
        &self,
        source: &[u8],
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        source_layout
            .validate_bytes(source.len())
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if source_layout.element_type() != destination.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference upload layout differs from destination element type",
            ));
        }
        let length = source_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let end = destination_offset_bytes
            .checked_add(length)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract("reference upload range overflows")
            })?;
        let destination = destination.region(destination_offset_bytes..end)?;
        Ok(ReferenceDeviceCommand::transfer(
            HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Upload {
                source: source.to_vec().into_boxed_slice(),
                destination,
            },
        ))
    }

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        let end = destination_offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract("reference zero range overflows")
            })?;
        let destination = destination.region(destination_offset_bytes..end)?;
        Ok(ReferenceDeviceCommand::transfer(
            DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Zero { destination },
        ))
    }

    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract(
                    "reference stream belongs to another runtime",
                ),
            ));
        }
        if commands.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract(
                    "reference runtime cannot submit an empty batch",
                ),
            ));
        }
        validate_submission_requirements(
            commands.timing_mode(),
            commands.compute_path_requirement(),
            commands.reusable_execution_capture().is_some(),
        )
        .map_err(DefinitelyNotSubmitted::new)?;
        let attribution_required = commands
            .attribution_requirement()
            .logical_execution_path_required();
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
            .collect::<Result<Vec<_>, ReferenceDeviceRuntimeError>>()
            .map_err(DefinitelyNotSubmitted::new)?;
        for (_, _, command) in &entries {
            command
                .validate_runtime(self.runtime_instance)
                .map_err(DefinitelyNotSubmitted::new)?;
        }
        let attribution = if attribution_required {
            let rows = entries
                .iter()
                .enumerate()
                .map(|(command_index, (phase, node_index, command))| {
                    let command_index = u32::try_from(command_index).map_err(|_| {
                        ReferenceDeviceRuntimeError::contract("reference command index exceeds u32")
                    })?;
                    command.attribution(command_index, *node_index, *phase)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(DefinitelyNotSubmitted::new)?;
            Some(DeviceSubmissionAttribution::new(rows).ok_or_else(|| {
                DefinitelyNotSubmitted::new(ReferenceDeviceRuntimeError::contract(
                    "reference submission attribution is empty or unordered",
                ))
            })?)
        } else {
            None
        };
        for (_, _, command) in &entries {
            command.execute(&self.counters);
        }
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
        self.counters.commands.fetch_add(
            u64::try_from(entries.len()).expect("reference command count fits u64"),
            Ordering::Relaxed,
        );
        stream.state = StreamState::Submitted;
        let fence_instance = NEXT_FENCE_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("reference fence identity space exhausted");
        Ok(ReferenceDeviceFence {
            runtime_instance: self.runtime_instance,
            fence_instance,
            attribution,
        })
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        (fence.runtime_instance == self.runtime_instance)
            .then(|| fence.attribution.clone())
            .flatten()
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        if fence.runtime_instance != self.runtime_instance {
            return FenceQuery::Indeterminate(ReferenceDeviceRuntimeError::contract(
                "reference fence belongs to another runtime",
            ));
        }
        FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        if fence.runtime_instance != self.runtime_instance {
            return Err(FenceIndeterminate::new(
                ReferenceDeviceRuntimeError::contract("reference fence belongs to another runtime"),
            ));
        }
        Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference stream belongs to another runtime",
            ));
        }
        stream.state = StreamState::Ready;
        Ok(())
    }

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        if stream.runtime_instance != self.runtime_instance
            || source.runtime_instance != self.runtime_instance
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback belongs to another runtime",
            ));
        }
        if output_layout.element_type() != source.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback layout differs from source element type",
            ));
        }
        let output_bytes = output_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let length = region.length_bytes();
        let source_end = region
            .source_offset_bytes()
            .checked_add(length)
            .filter(|end| *end <= source.descriptor.size_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference readback source exceeds its admitted allocation",
                )
            })?;
        let output_end = region
            .destination_offset_bytes()
            .checked_add(length)
            .filter(|end| *end <= output_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference readback destination exceeds its host layout",
                )
            })?;
        self.synchronize(stream)?;
        let source_bytes = source
            .region(region.source_offset_bytes()..source_end)?
            .read();
        let output_len = usize::try_from(output_bytes).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback output exceeds usize")
        })?;
        let output_start = usize::try_from(region.destination_offset_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback offset exceeds usize")
        })?;
        let output_end = usize::try_from(output_end).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback end exceeds usize")
        })?;
        if source_bytes.len() != output_end - output_start {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback source and destination lengths differ",
            ));
        }
        let mut output = vec![0_u8; output_len];
        output[output_start..output_end].copy_from_slice(&source_bytes);
        self.counters
            .readback_bytes
            .fetch_add(length, Ordering::Relaxed);
        Ok(output)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("reference_runtime", error.to_string(), false)
    }
}

fn validate_submission_requirements(
    timing_mode: DeviceTimingMode,
    compute_path: DeviceComputePathRequirement,
    has_reusable_capture: bool,
) -> Result<(), ReferenceDeviceRuntimeError> {
    if timing_mode != DeviceTimingMode::Off {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime does not provide device timing evidence",
        ));
    }
    if matches!(
        compute_path,
        DeviceComputePathRequirement::ReplayedOnly
            | DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
    ) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime cannot satisfy a replay-required compute submission",
        ));
    }
    if has_reusable_capture {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime cannot consume reusable execution capture metadata",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::reference::composition::reference_vnext_runtime_config;
    use ferrum_interfaces::vnext::{BufferUsage, DeviceId, ResourceId};

    fn config() -> ReferenceDeviceRuntimeConfig {
        reference_vnext_runtime_config(
            DeviceId::new("device.reference.runtime-test").expect("valid device id"),
        )
        .expect("valid reference config")
    }

    fn buffer(
        runtime: &ReferenceDeviceRuntime,
        resource_id: &str,
        contents: &[u8],
        alignment_bytes: usize,
        element_type: ElementType,
    ) -> ReferenceDeviceBuffer {
        let (bytes, logical_offset_bytes) =
            aligned_storage(contents.len(), alignment_bytes).expect("aligned test storage");
        runtime
            .counters
            .live_allocations
            .fetch_add(1, Ordering::Relaxed);
        let buffer = ReferenceDeviceBuffer {
            descriptor: BufferDescriptor {
                resource_id: ResourceId::new(resource_id).expect("valid test resource id"),
                size_bytes: contents.len() as u64,
                alignment_bytes: alignment_bytes as u64,
                usage: BufferUsage::Transfer,
                element_type,
            },
            runtime_instance: runtime.runtime_instance,
            allocation: Arc::new(ReferenceAllocation {
                bytes: Mutex::new(bytes),
                logical_offset_bytes,
                live_allocations: Arc::clone(&runtime.counters.live_allocations),
            }),
        };
        buffer
            .region(0..contents.len() as u64)
            .expect("test buffer region")
            .write(contents);
        buffer
    }

    #[test]
    fn aligned_reference_storage_fulfills_descriptor_contract() {
        for alignment in [1, 2, 16, 64, 4096] {
            let (bytes, offset) = aligned_storage(257, alignment).expect("aligned storage");
            assert_eq!((bytes.as_ptr() as usize + offset) % alignment, 0);
            assert!(offset + 257 <= bytes.len());
        }
    }

    #[test]
    fn descriptor_cannot_overclaim_device_class_or_capabilities() {
        ReferenceDeviceRuntime::new(config()).expect("fixed reference descriptor must be valid");

        let mut wrong_class = config();
        wrong_class.descriptor.class = DeviceClass::Accelerator;
        assert!(ReferenceDeviceRuntime::new(wrong_class).is_err());

        let mut extra_capability = config();
        extra_capability.descriptor.capabilities.insert(
            CapabilityId::new("capability.reference.unimplemented")
                .expect("valid synthetic capability id"),
        );
        assert!(ReferenceDeviceRuntime::new(extra_capability).is_err());
    }

    #[test]
    fn submission_requirements_fail_closed_before_reference_execution() {
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::Adaptive,
            false,
        )
        .is_ok());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::EagerOnly,
            false,
        )
        .is_ok());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::ReplayedOnly,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Completion,
            DeviceComputePathRequirement::Adaptive,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::Adaptive,
            true,
        )
        .is_err());
    }

    #[test]
    fn foreign_stream_state_fails_closed() {
        let first = ReferenceDeviceRuntime::new(config()).expect("first runtime");
        let second = ReferenceDeviceRuntime::new(config()).expect("second runtime");
        let stream = first.create_stream().expect("first stream");
        assert_eq!(first.stream_state(&stream), StreamState::Ready);
        assert_eq!(second.stream_state(&stream), StreamState::Failed);
    }

    #[test]
    fn readback_honors_host_offset_and_element_type() {
        let runtime = ReferenceDeviceRuntime::new(config()).expect("reference runtime");
        let source = buffer(
            &runtime,
            "resource.reference.readback",
            &[10, 11, 12, 13],
            16,
            ElementType::U8,
        );
        let mut stream = runtime.create_stream().expect("reference stream");
        let region = CopyRegion::new(1, 2, 2).expect("valid readback region");
        let output = runtime
            .readback(
                &mut stream,
                &source,
                region,
                HostTransferLayout::new(ElementType::U8, 6).expect("valid host layout"),
            )
            .expect("offset readback");
        assert_eq!(output, [0, 0, 11, 12, 0, 0]);
        assert!(runtime
            .readback(
                &mut stream,
                &source,
                region,
                HostTransferLayout::new(ElementType::F16, 3).expect("valid mismatched layout"),
            )
            .is_err());
    }
}
