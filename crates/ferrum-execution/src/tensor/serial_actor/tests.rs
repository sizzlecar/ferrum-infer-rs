//! Ownership/poison tests only; real tensor numerics and fresh request state
//! are exercised by the ReferenceDeviceRuntime component integration tests.
use super::*;
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct StreamObservations {
    created: AtomicUsize,
    dropped: AtomicUsize,
}

struct Stream(Arc<StreamObservations>);
impl Drop for Stream {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct Runtime {
    descriptor: DeviceDescriptor,
    streams: Arc<StreamObservations>,
}

fn runtime() -> Arc<Runtime> {
    Arc::new(Runtime {
        descriptor: DeviceDescriptor {
            id: DeviceId::new("device.serial-actor-lifetime").unwrap(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 4096,
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        },
        streams: Arc::new(StreamObservations::default()),
    })
}

impl DeviceRuntime for Runtime {
    type Buffer = BufferDescriptor;
    type Stream = Stream;
    type Command = ();
    type Fence = ();
    type Error = std::io::Error;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn create_stream(&self) -> Result<Stream, Self::Error> {
        self.streams.created.fetch_add(1, Ordering::SeqCst);
        Ok(Stream(self.streams.clone()))
    }
    fn stream_state(&self, _: &Stream) -> StreamState {
        StreamState::Ready
    }
    fn allocate(&self, _: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        unreachable!("lifetime fixture does not allocate tensor storage")
    }
    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.clone()
    }
    fn encode_copy(
        &self,
        _: &Self::Buffer,
        _: &Self::Buffer,
        _: CopyRegion,
    ) -> Result<(), Self::Error> {
        unreachable!("lifetime fixture does not encode device work")
    }
    fn encode_upload(
        &self,
        _: &[u8],
        _: HostTransferLayout,
        _: &Self::Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        unreachable!("lifetime fixture does not upload")
    }
    fn encode_zero(&self, _: &Self::Buffer, _: u64, _: u64) -> Result<(), Self::Error> {
        unreachable!("lifetime fixture does not zero storage")
    }
    fn submit(
        &self,
        _: &mut Stream,
        _: DeviceCommandBatch<()>,
    ) -> Result<(), DefinitelyNotSubmitted<Self::Error>> {
        unreachable!("lifetime fixture does not submit")
    }
    fn query_fence(&self, _: &()) -> FenceQuery<Self::Error> {
        unreachable!("lifetime fixture creates no fence")
    }
    fn wait_fence(
        &self,
        _: &(),
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        unreachable!("lifetime fixture creates no fence")
    }
    fn synchronize(&self, _: &mut Stream) -> Result<(), Self::Error> {
        unreachable!("actor ownership must not add a synchronization")
    }
    fn readback(
        &self,
        _: &mut Stream,
        _: &Self::Buffer,
        _: CopyRegion,
        _: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        unreachable!("lifetime fixture does not read tensors")
    }
    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("serial_actor_lifetime", error.to_string(), false)
    }
}

#[test]
fn successful_calls_keep_one_lane_until_its_last_owner_retires() {
    let runtime = runtime();
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    assert!(!actor.is_failed());
    let first = actor.execute(|lane| Ok(lane.id())).unwrap();
    let retained = actor.execute(|lane| Ok(lane.clone())).unwrap();
    assert_eq!(first, retained.id());
    assert!(!actor.is_failed());
    assert_eq!(runtime.streams.created.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.streams.dropped.load(Ordering::SeqCst), 0);
    drop(actor);
    assert_eq!(runtime.streams.dropped.load(Ordering::SeqCst), 0);
    drop(retained);
    assert_eq!(runtime.streams.dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn error_and_unwind_stop_actor_without_revoking_external_lane_ownership() {
    for unwind in [false, true] {
        let runtime = runtime();
        let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
        let retained = actor.execute(|lane| Ok(lane.clone())).unwrap();
        if unwind {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = actor.execute::<()>(|_| panic!("injected actor unwind"));
            }))
            .is_err());
        } else {
            assert!(actor
                .execute::<()>(|_| Err(invalid("injected actor failure").into()))
                .is_err());
        }
        assert!(actor.is_failed());
        assert!(actor
            .execute::<()>(|_| panic!("failed actor must not enter another operation"))
            .is_err());
        assert_eq!(runtime.streams.created.load(Ordering::SeqCst), 1);
        drop(actor);
        assert_eq!(runtime.streams.dropped.load(Ordering::SeqCst), 0);
        drop(retained);
        assert_eq!(runtime.streams.dropped.load(Ordering::SeqCst), 1);
    }
}
