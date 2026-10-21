//! Only completion observations and managed readback fail here. Allocation,
//! uploads, computation, submission and attribution use the real Reference runtime.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use ferrum_kernels::backend::reference::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceDeviceStream,
};
use ferrum_runtime::*;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FenceMode {
    Real,
    Indeterminate,
    Pending,
    FailedButQuiescent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadbackMode {
    Real,
    ErrorAfterWrite,
    PanicAfterWrite,
}

#[derive(Debug)]
pub(super) enum FaultError {
    Reference(ReferenceDeviceRuntimeError),
    Injected(&'static str),
}

impl fmt::Display for FaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference(error) => error.fmt(f),
            Self::Injected(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for FaultError {}
impl From<ReferenceDeviceRuntimeError> for FaultError {
    fn from(error: ReferenceDeviceRuntimeError) -> Self {
        Self::Reference(error)
    }
}

pub(super) struct FaultRuntime {
    pub(super) inner: Arc<ReferenceDeviceRuntime>,
    descriptor: DeviceDescriptor,
    fence: Mutex<FenceMode>,
    readback: Mutex<ReadbackMode>,
    synchronize_fails: AtomicBool,
    pub(super) readback_calls: AtomicUsize,
}

impl FaultRuntime {
    pub(super) fn new(inner: Arc<ReferenceDeviceRuntime>) -> Self {
        let mut descriptor = inner.descriptor().clone();
        descriptor.runtime_implementation_fingerprint = format!(
            "{:x}",
            Sha256::digest(
                [
                    descriptor.runtime_implementation_fingerprint.as_bytes(),
                    include_bytes!("fault_runtime.rs")
                ]
                .concat()
            )
        );
        Self {
            inner,
            descriptor,
            fence: Mutex::new(FenceMode::Real),
            readback: Mutex::new(ReadbackMode::Real),
            synchronize_fails: AtomicBool::new(false),
            readback_calls: AtomicUsize::new(0),
        }
    }
    pub(super) fn set_fence(&self, mode: FenceMode) {
        *self.fence.lock().unwrap() = mode;
    }
    pub(super) fn set_readback(&self, mode: ReadbackMode) {
        *self.readback.lock().unwrap() = mode;
    }
    pub(super) fn set_synchronize_fails(&self, value: bool) {
        self.synchronize_fails.store(value, Ordering::SeqCst);
    }
}

fn terminal(
    receipt: DeviceTerminalReceipt<ReferenceDeviceRuntimeError>,
) -> DeviceTerminalReceipt<FaultError> {
    let (status, execution, submission) = receipt.into_parts();
    let status = match status {
        DeviceTerminal::Succeeded => DeviceTerminal::Succeeded,
        DeviceTerminal::FailedButQuiescent(error) => {
            DeviceTerminal::FailedButQuiescent(error.into())
        }
    };
    DeviceTerminalReceipt::profiled_with_submission_timing(status, execution, submission)
}

impl DeviceRuntime for FaultRuntime {
    type Buffer = ReferenceDeviceBuffer;
    type Stream = ReferenceDeviceStream;
    type Command = ReferenceDeviceCommand;
    type Fence = ReferenceDeviceFence;
    type Error = FaultError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        self.inner.attention_execution_policy()
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        self.inner.allocate(permit).map_err(Into::into)
    }
    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        self.inner.buffer_descriptor(buffer)
    }
    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        self.inner.create_stream().map_err(Into::into)
    }
    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        self.inner.stream_state(stream)
    }
    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        self.inner
            .encode_copy(source, destination, region)
            .map_err(Into::into)
    }
    fn encode_upload(
        &self,
        source: &[u8],
        layout: HostTransferLayout,
        destination: &Self::Buffer,
        offset: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.inner
            .encode_upload(source, layout, destination, offset)
            .map_err(Into::into)
    }
    fn supports_managed_tensor_io(&self) -> bool {
        self.inner.supports_managed_tensor_io()
    }
    fn encode_managed_upload(
        &self,
        source: ManagedHostBytes,
        layout: HostTransferLayout,
        destination: &Self::Buffer,
        offset: u64,
    ) -> Result<Option<Self::Command>, Self::Error> {
        self.inner
            .encode_managed_upload(source, layout, destination, offset)
            .map_err(Into::into)
    }
    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        offset: u64,
        length: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.inner
            .encode_zero(destination, offset, length)
            .map_err(Into::into)
    }
    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        self.inner
            .submit(stream, commands)
            .map_err(|error| DefinitelyNotSubmitted::new(error.into_error().into()))
    }
    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        self.inner.submission_attribution(fence)
    }
    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        match *self.fence.lock().unwrap() {
            FenceMode::Pending => FenceQuery::Pending,
            FenceMode::Indeterminate => {
                FenceQuery::Indeterminate(FaultError::Injected("injected indeterminate fence"))
            }
            FenceMode::FailedButQuiescent => FenceQuery::Terminal(
                DeviceTerminalReceipt::unprofiled(DeviceTerminal::FailedButQuiescent(
                    FaultError::Injected("injected terminal failure"),
                )),
            ),
            FenceMode::Real => match self.inner.query_fence(fence) {
                FenceQuery::Pending => FenceQuery::Pending,
                FenceQuery::Indeterminate(error) => FenceQuery::Indeterminate(error.into()),
                FenceQuery::Terminal(receipt) => FenceQuery::Terminal(terminal(receipt)),
            },
        }
    }
    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        match *self.fence.lock().unwrap() {
            // The wait API cannot return Pending. Nontermination is reported
            // as indeterminate; the nonblocking query proves Pending separately.
            FenceMode::Pending | FenceMode::Indeterminate => Err(FenceIndeterminate::new(
                FaultError::Injected("injected indeterminate wait"),
            )),
            FenceMode::FailedButQuiescent => Ok(DeviceTerminalReceipt::unprofiled(
                DeviceTerminal::FailedButQuiescent(FaultError::Injected(
                    "injected terminal failure",
                )),
            )),
            FenceMode::Real => self
                .inner
                .wait_fence(fence)
                .map(terminal)
                .map_err(|error| FenceIndeterminate::new(error.into_error().into())),
        }
    }
    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        if self.synchronize_fails.load(Ordering::SeqCst) {
            return Err(FaultError::Injected("injected synchronization failure"));
        }
        self.inner.synchronize(stream).map_err(Into::into)
    }
    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        self.inner
            .readback(stream, source, region, layout)
            .map_err(Into::into)
    }
    fn readback_managed_into(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
        destination: &mut ManagedHostBuffer,
    ) -> Result<bool, Self::Error> {
        self.readback_calls.fetch_add(1, Ordering::SeqCst);
        let mode = *self.readback.lock().unwrap();
        match mode {
            ReadbackMode::Real => self
                .inner
                .readback_managed_into(stream, source, region, layout, destination)
                .map_err(Into::into),
            ReadbackMode::ErrorAfterWrite => {
                destination.as_mut()[0] = 0xee;
                Err(FaultError::Injected(
                    "injected readback error after partial write",
                ))
            }
            ReadbackMode::PanicAfterWrite => {
                destination.as_mut()[0] = 0xdd;
                panic!("injected readback panic after partial write")
            }
        }
    }
    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("reference_fault_observer", error.to_string(), false)
    }
}
