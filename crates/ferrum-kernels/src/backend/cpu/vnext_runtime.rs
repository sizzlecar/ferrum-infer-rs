//! Host execution for the production plan runtime. CPU commands run in order
//! under a runtime-wide submission lock; errors after execution starts produce
//! failed, quiescent fences, never permission to retry a partially written batch.

use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ferrum_interfaces::vnext::{
    BufferDescriptor, CopyRegion, DefinitelyNotSubmitted, DeviceAllocationPermit, DeviceClass,
    DeviceCommandBatch, DeviceCommandPhase, DeviceComputePathRequirement, DeviceDescriptor,
    DeviceErrorReport, DeviceRuntime, DeviceSubmissionAttribution, DeviceTerminal,
    DeviceTerminalReceipt, DeviceTimingMeasurement, DeviceTimingMode,
    DeviceTimingUnavailableReason, FenceIndeterminate, FenceQuery, HostTransferLayout, StreamState,
    VNextError, DEVICE_COPY_NATIVE_OPERATION_ID, DEVICE_ZERO_NATIVE_OPERATION_ID,
    HOST_UPLOAD_NATIVE_OPERATION_ID,
};

mod command;
mod host_memory;
mod memory;

use command::CommandKind;
pub use command::CpuDeviceCommand;
pub(crate) use command::CpuKernelLaunch;
pub(crate) use host_memory::host_memory_capacity;
pub(crate) use memory::CpuBufferRegion;
pub use memory::CpuDeviceBuffer;
pub(crate) use memory::CpuRegionSet;
use memory::{MemoryBudget, Storage};

static NEXT_RUNTIME_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuRuntimeError(String);

impl CpuRuntimeError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CpuRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CpuRuntimeError {}

impl From<VNextError> for CpuRuntimeError {
    fn from(error: VNextError) -> Self {
        Self::new(error.to_string())
    }
}

pub struct CpuDeviceStream {
    runtime_instance: u64,
    state: StreamState,
    failure: Option<CpuRuntimeError>,
}

pub struct CpuDeviceFence {
    runtime_instance: u64,
    failure: Option<CpuRuntimeError>,
    attribution: Option<DeviceSubmissionAttribution>,
    timing_mode: DeviceTimingMode,
}

impl CpuDeviceFence {
    fn receipt(&self) -> DeviceTerminalReceipt<CpuRuntimeError> {
        let terminal = match &self.failure {
            Some(error) => DeviceTerminal::FailedButQuiescent(error.clone()),
            None => DeviceTerminal::Succeeded,
        };
        // Host execution still produces an exact completion fence. Missing
        // device clocks are unavailable observations, never a compute failure
        // or a reason to label a host wall-clock duration as device time.
        DeviceTerminalReceipt::profiled_with_submission_timing(
            terminal,
            if self.timing_mode.completion_enabled() {
                DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendUnsupported,
                )
            } else {
                DeviceTimingMeasurement::NotRequested
            },
            if self.timing_mode.physical_span_attribution_enabled() {
                DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendUnsupported,
                )
            } else {
                DeviceTimingMeasurement::NotRequested
            },
        )
    }
}

pub struct CpuDeviceRuntime {
    descriptor: DeviceDescriptor,
    runtime_instance: u64,
    budget: Arc<MemoryBudget>,
    execution: Mutex<()>,
}

impl CpuDeviceRuntime {
    // Only the CPU composition may create a production descriptor. Keep the
    // public product constructor there, alongside its exact capability registry.
    pub(crate) fn new(descriptor: DeviceDescriptor) -> Result<Self, CpuRuntimeError> {
        descriptor.validate()?;
        if descriptor.class != DeviceClass::Host || descriptor.ordinal != 0 {
            return Err(CpuRuntimeError::new(
                "CPU runtime requires the host device descriptor",
            ));
        }
        let runtime_instance = NEXT_RUNTIME_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| CpuRuntimeError::new("CPU runtime identity exhausted"))?;
        Ok(Self {
            budget: MemoryBudget::new(descriptor.total_memory_bytes),
            descriptor,
            runtime_instance,
            execution: Mutex::new(()),
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        self.budget.used()
    }
    pub fn peak_resident_bytes(&self) -> u64 {
        self.budget.peak()
    }

    fn validate_buffer(&self, buffer: &CpuDeviceBuffer) -> Result<(), CpuRuntimeError> {
        if buffer.runtime_instance == self.runtime_instance {
            Ok(())
        } else {
            Err(CpuRuntimeError::new(
                "CPU buffer belongs to another runtime",
            ))
        }
    }

    fn validate_stream(&self, stream: &CpuDeviceStream) -> Result<(), CpuRuntimeError> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(CpuRuntimeError::new(
                "CPU stream belongs to another runtime",
            ));
        }
        if let Some(error) = &stream.failure {
            return Err(error.clone());
        }
        Ok(())
    }

    fn execute_entries(
        &self,
        stream: &mut CpuDeviceStream,
        entries: Vec<(DeviceCommandPhase, Option<u32>, CpuDeviceCommand)>,
        attribution_required: bool,
        timing_mode: DeviceTimingMode,
    ) -> Result<CpuDeviceFence, DefinitelyNotSubmitted<CpuRuntimeError>> {
        self.validate_stream(stream)
            .map_err(DefinitelyNotSubmitted::new)?;
        if entries.is_empty() {
            return Err(DefinitelyNotSubmitted::new(CpuRuntimeError::new(
                "CPU submission is empty",
            )));
        }
        // Every command is validated before any command is executed.
        for (_, _, command) in &entries {
            command
                .validate_runtime(self.runtime_instance)
                .map_err(DefinitelyNotSubmitted::new)?;
        }
        let attribution = if attribution_required {
            let rows = entries
                .iter()
                .enumerate()
                .map(|(index, (phase, node, command))| {
                    let index = u32::try_from(index)
                        .map_err(|_| CpuRuntimeError::new("CPU command index exceeds u32"))?;
                    command.attribution(index, *node, *phase)
                })
                .collect::<Result<Vec<_>, CpuRuntimeError>>()
                .map_err(DefinitelyNotSubmitted::new)?;
            Some(DeviceSubmissionAttribution::new(rows).ok_or_else(|| {
                DefinitelyNotSubmitted::new(CpuRuntimeError::new(
                    "invalid CPU submission attribution",
                ))
            })?)
        } else {
            None
        };
        let _execution = self.execution.lock().map_err(|_| {
            DefinitelyNotSubmitted::new(CpuRuntimeError::new("CPU execution lock was poisoned"))
        })?;
        stream.state = StreamState::Submitted;
        let result = catch_unwind(AssertUnwindSafe(|| {
            for (_, _, command) in &entries {
                command.execute()?;
            }
            Ok::<(), CpuRuntimeError>(())
        }))
        .unwrap_or_else(|_| {
            Err(CpuRuntimeError::new(
                "CPU command panicked; submission is quiescent and its stream is failed",
            ))
        });
        let failure = result.err();
        if let Some(error) = &failure {
            stream.state = StreamState::Failed;
            stream.failure = Some(error.clone());
        }
        // Synchronous execution is now quiescent. Do not claim full-batch work
        // attribution if execution stopped partway through a provider command.
        Ok(CpuDeviceFence {
            runtime_instance: self.runtime_instance,
            attribution: if failure.is_none() { attribution } else { None },
            failure,
            timing_mode,
        })
    }
}

impl DeviceRuntime for CpuDeviceRuntime {
    type Buffer = CpuDeviceBuffer;
    type Stream = CpuDeviceStream;
    type Command = CpuDeviceCommand;
    type Fence = CpuDeviceFence;
    type Error = CpuRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        CpuDeviceBuffer::allocate(
            BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
            self.runtime_instance,
            &self.budget,
        )
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(CpuDeviceStream {
            runtime_instance: self.runtime_instance,
            state: StreamState::Ready,
            failure: None,
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
        self.validate_buffer(source)?;
        self.validate_buffer(destination)?;
        region.validate_bounds(&source.descriptor, &destination.descriptor)?;
        if source.descriptor.element_type != destination.descriptor.element_type {
            return Err(CpuRuntimeError::new("CPU copy element types differ"));
        }
        let source = source.region(
            region.source_offset_bytes()..region.source_offset_bytes() + region.length_bytes(),
        )?;
        let destination = destination.region(
            region.destination_offset_bytes()
                ..region.destination_offset_bytes() + region.length_bytes(),
        )?;
        Ok(CpuDeviceCommand::transfer(
            DEVICE_COPY_NATIVE_OPERATION_ID.as_str(),
            CommandKind::Copy {
                source,
                destination,
            },
        ))
    }

    fn encode_upload(
        &self,
        source: &[u8],
        layout: HostTransferLayout,
        destination: &Self::Buffer,
        offset: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.validate_buffer(destination)?;
        layout.validate_bytes(source.len())?;
        if layout.element_type() != destination.descriptor.element_type {
            return Err(CpuRuntimeError::new("CPU upload element types differ"));
        }
        let size = layout.byte_len()?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| CpuRuntimeError::new("CPU upload range overflows"))?;
        let destination = destination.region(offset..end)?;
        let mut owned = Storage::new(size, 1, &self.budget)?;
        owned.bytes_mut().copy_from_slice(source);
        Ok(CpuDeviceCommand::transfer(
            HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
            CommandKind::Upload {
                source: owned,
                destination,
            },
        ))
    }

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.validate_buffer(destination)?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| CpuRuntimeError::new("CPU zero range overflows"))?;
        let destination = destination.region(offset..end)?;
        Ok(CpuDeviceCommand::transfer(
            DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
            CommandKind::Zero { destination },
        ))
    }

    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        let timing_mode = commands.timing_mode();
        validate_submission_requirements(
            commands.compute_path_requirement(),
            commands
                .reusable_execution_capture()
                .map(|capture| (capture.node_count(), capture.eager_boundary_node_indices())),
        )
        .map_err(DefinitelyNotSubmitted::new)?;
        let attribution = commands
            .attribution_requirement()
            .logical_execution_path_required();
        let entries = commands
            .into_entries()
            .into_iter()
            .map(|entry| {
                let (phase, node, work, command) = entry.into_parts();
                let command = match work {
                    Some(work) => command.bind_logical_work(work)?,
                    None => command,
                };
                Ok((phase, node, command))
            })
            .collect::<Result<Vec<_>, CpuRuntimeError>>()
            .map_err(DefinitelyNotSubmitted::new)?;
        self.execute_entries(stream, entries, attribution, timing_mode)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        (fence.runtime_instance == self.runtime_instance)
            .then(|| fence.attribution.clone())
            .flatten()
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        if fence.runtime_instance != self.runtime_instance {
            return FenceQuery::Indeterminate(CpuRuntimeError::new(
                "CPU fence belongs to another runtime",
            ));
        }
        FenceQuery::Terminal(fence.receipt())
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        if fence.runtime_instance != self.runtime_instance {
            return Err(FenceIndeterminate::new(CpuRuntimeError::new(
                "CPU fence belongs to another runtime",
            )));
        }
        Ok(fence.receipt())
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        self.validate_stream(stream)?;
        let _execution = self
            .execution
            .lock()
            .map_err(|_| CpuRuntimeError::new("CPU execution lock was poisoned"))?;
        stream.state = StreamState::Ready;
        Ok(())
    }

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        self.validate_buffer(source)?;
        self.validate_stream(stream)?;
        if layout.element_type() != source.descriptor.element_type {
            return Err(CpuRuntimeError::new("CPU readback element types differ"));
        }
        let output_size = layout.byte_len()?;
        let output_end = region
            .destination_offset_bytes()
            .checked_add(region.length_bytes())
            .filter(|end| *end <= output_size)
            .ok_or_else(|| CpuRuntimeError::new("CPU readback exceeds host destination layout"))?;
        let source_end = region
            .source_offset_bytes()
            .checked_add(region.length_bytes())
            .ok_or_else(|| CpuRuntimeError::new("CPU readback source range overflows"))?;
        let source = source.region(region.source_offset_bytes()..source_end)?;
        let output_len = usize::try_from(output_size)
            .map_err(|_| CpuRuntimeError::new("CPU readback exceeds usize"))?;
        let output_start = usize::try_from(region.destination_offset_bytes())
            .map_err(|_| CpuRuntimeError::new("CPU readback offset exceeds usize"))?;
        let output_end = usize::try_from(output_end)
            .map_err(|_| CpuRuntimeError::new("CPU readback end exceeds usize"))?;
        let _reservation = self.budget.reserve(output_size)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|error| CpuRuntimeError::new(error.to_string()))?;
        output.resize(output_len, 0);
        let _execution = self
            .execution
            .lock()
            .map_err(|_| CpuRuntimeError::new("CPU execution lock was poisoned"))?;
        source.with_read(|bytes| output[output_start..output_end].copy_from_slice(bytes))?;
        stream.state = StreamState::Ready;
        Ok(output)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("cpu_runtime", error.to_string(), false)
    }
}

fn validate_submission_requirements(
    path: DeviceComputePathRequirement,
    capture: Option<(u32, &[u32])>,
) -> Result<(), CpuRuntimeError> {
    if matches!(
        path,
        DeviceComputePathRequirement::ReplayedOnly
            | DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
    ) {
        return Err(CpuRuntimeError::new(
            "CPU runtime supports eager execution only",
        ));
    }
    // Core attaches this metadata to the full eager encoding even when every
    // provider declares an eager boundary. It is not a replay requirement.
    // Accept that topology without publishing a reusable executable catalog.
    if let Some((nodes, boundaries)) = capture {
        if nodes == 0 || !boundaries.iter().copied().eq(0..nodes) {
            return Err(CpuRuntimeError::new(
                "CPU capture topology contains a node that was not declared eager",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
