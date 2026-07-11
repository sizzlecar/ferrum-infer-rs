use ferrum_interfaces::vnext::*;
use std::collections::BTreeSet;
use std::io;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Policy {
    dynamic_storage_profile_order: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for Policy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }

    fn memory_capacity_bytes(&self) -> u64 {
        1024
    }

    fn memory_reserve_bytes(&self) -> u64 {
        128
    }

    fn maximum_active_sequences(&self) -> u32 {
        4
    }

    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.dynamic_storage_profile_order
    }

    fn validate(&self) -> Result<(), VNextError> {
        Ok(())
    }
}

struct Buffer;
struct Stream;
struct Command;
struct Fence;

struct Runtime {
    descriptor: DeviceDescriptor,
}

impl DeviceRuntime for Runtime {
    type Buffer = Buffer;
    type Stream = Stream;
    type Command = Command;
    type Fence = Fence;
    type Error = io::Error;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(&self, _permit: DeviceAllocationPermit<'_>) -> Result<Buffer, io::Error> {
        Ok(Buffer)
    }

    fn buffer_descriptor(&self, _buffer: &Buffer) -> BufferDescriptor {
        BufferDescriptor {
            resource_id: ResourceId::new("resource/buffer").unwrap(),
            size_bytes: 4,
            alignment_bytes: 4,
            usage: BufferUsage::Activations,
            element_type: ElementType::F32,
        }
    }

    fn create_stream(&self) -> Result<Stream, io::Error> {
        Ok(Stream)
    }

    fn stream_state(&self, _stream: &Stream) -> StreamState {
        StreamState::Ready
    }

    fn encode_copy(
        &self,
        _source: &Buffer,
        _destination: &Buffer,
        _region: CopyRegion,
    ) -> Result<Command, io::Error> {
        Ok(Command)
    }

    fn encode_upload(
        &self,
        _source: &[u8],
        _source_layout: HostTransferLayout,
        _destination: &Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Command, io::Error> {
        Ok(Command)
    }

    fn encode_zero(
        &self,
        _destination: &Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Command, io::Error> {
        Ok(Command)
    }

    fn submit(
        &self,
        _stream: &mut Stream,
        _command: Command,
    ) -> Result<Fence, DefinitelyNotSubmitted<io::Error>> {
        Ok(Fence)
    }

    fn query_fence(&self, _fence: &Fence) -> FenceQuery<io::Error> {
        FenceQuery::Terminal(DeviceTerminal::Succeeded)
    }

    fn wait_fence(
        &self,
        _fence: &Fence,
    ) -> Result<DeviceTerminal<io::Error>, FenceIndeterminate<io::Error>> {
        Ok(DeviceTerminal::Succeeded)
    }

    fn synchronize(&self, _stream: &mut Stream) -> Result<(), io::Error> {
        Ok(())
    }

    fn readback(
        &self,
        _stream: &mut Stream,
        _source: &Buffer,
        _region: CopyRegion,
        _output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, io::Error> {
        Ok(vec![0; 4])
    }

    fn describe_error(&self, _error: &io::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("device_error", "reference device error", false)
    }
}

fn assert_object_safe(
    _runtime: &dyn DeviceRuntime<
        Buffer = Buffer,
        Stream = Stream,
        Command = Command,
        Fence = Fence,
        Error = io::Error,
    >,
) {
}

fn assert_other_dyn_boundaries(
    _operation: &dyn OperationContract,
    _provider: &dyn OperationProvider<Runtime>,
    _planner: &dyn ExecutionPlanner<Policy = Policy>,
    _events: &dyn ExecutionEventSink,
    _family: &dyn ModelFamilyRegistration,
) {
}

fn assert_owned_root_api(root: Arc<PlanRuntimeResources<Runtime>>) {
    let binding = root.trusted_runtime_binding().unwrap();
    drop(binding);
    let close = PlanRuntimeResources::close(root);
    drop(close);
}

fn assert_send_sync<T: Send + Sync>() {}

fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

fn main() {
    let runtime = Runtime {
        descriptor: DeviceDescriptor {
            id: DeviceId::new("device/reference/0").unwrap(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1024,
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
    };
    assert_object_safe(&runtime);
    assert_send_sync::<PlanRuntimeResources<Runtime>>();
    let _ = assert_other_dyn_boundaries;
    let _ = assert_owned_root_api;
}
