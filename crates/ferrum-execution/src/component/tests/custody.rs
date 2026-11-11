//! Actual component/resource lifecycle with a metadata-only device. These
//! tests exercise owner custody and runtime acknowledgements, not numerics.
use super::*;
use std::{io, panic::AssertUnwindSafe, sync::atomic::AtomicBool};

struct Runtime {
    inner: Arc<f::TestRuntime>,
    descriptor_panic: AtomicBool,
    managed_panic: AtomicBool,
    allocation_calls: AtomicUsize,
    unknown: AtomicBool,
    submitted: AtomicBool,
}
impl Runtime {
    fn new(plan: &ExecutionPlan) -> Arc<Self> {
        Arc::new(Self {
            inner: f::runtime(plan),
            descriptor_panic: AtomicBool::new(false),
            managed_panic: AtomicBool::new(false),
            allocation_calls: AtomicUsize::new(0),
            unknown: AtomicBool::new(false),
            submitted: AtomicBool::new(false),
        })
    }
}
impl DeviceRuntime for Runtime {
    type Buffer = f::TestBuffer;
    type Stream = ();
    type Command = ();
    type Fence = ();
    type Error = io::Error;
    fn descriptor(&self) -> &DeviceDescriptor {
        assert!(
            !self.descriptor_panic.swap(false, Ordering::SeqCst),
            "pre-native descriptor panic"
        );
        self.inner.descriptor()
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> io::Result<Self::Buffer> {
        self.allocation_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.allocate(permit)
    }
    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        self.inner.buffer_descriptor(buffer)
    }
    fn create_stream(&self) -> io::Result<()> {
        Ok(())
    }
    fn stream_state(&self, _: &()) -> StreamState {
        if self.submitted.load(Ordering::SeqCst) && self.unknown.load(Ordering::SeqCst) {
            StreamState::Submitted
        } else {
            StreamState::Ready
        }
    }
    fn encode_copy(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        region: CopyRegion,
    ) -> io::Result<()> {
        self.inner.encode_copy(a, b, region)
    }
    fn encode_upload(
        &self,
        _: &[u8],
        _: HostTransferLayout,
        _: &Self::Buffer,
        _: u64,
    ) -> io::Result<()> {
        panic!("managed lifecycle must not fall back to raw uploads")
    }
    fn supports_managed_tensor_io(&self) -> bool {
        assert!(
            !self.managed_panic.swap(false, Ordering::SeqCst),
            "constructor panic after root binding"
        );
        true
    }
    fn encode_managed_upload(
        &self,
        source: ManagedHostBytes,
        layout: HostTransferLayout,
        _: &Self::Buffer,
        _: u64,
    ) -> io::Result<Option<()>> {
        layout.validate_bytes(source.len()).unwrap();
        // No native DMA on this metadata device. The real core still holds its
        // upload owner through the actual synchronous completion protocol.
        Ok(Some(()))
    }
    fn encode_zero(&self, a: &Self::Buffer, offset: u64, bytes: u64) -> io::Result<()> {
        self.inner.encode_zero(a, offset, bytes)
    }
    fn submit(
        &self,
        _: &mut (),
        commands: DeviceCommandBatch<()>,
    ) -> Result<(), DefinitelyNotSubmitted<io::Error>> {
        assert!(!commands.is_empty());
        self.submitted.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn query_fence(&self, _: &()) -> FenceQuery<io::Error> {
        if self.unknown.load(Ordering::SeqCst) {
            FenceQuery::Pending
        } else {
            FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
        }
    }
    fn wait_fence(
        &self,
        _: &(),
    ) -> Result<DeviceTerminalReceipt<io::Error>, FenceIndeterminate<io::Error>> {
        if self.unknown.load(Ordering::SeqCst) {
            Err(FenceIndeterminate::new(io::Error::other(
                "original fence unknown",
            )))
        } else {
            self.submitted.store(false, Ordering::SeqCst);
            Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
        }
    }
    fn synchronize(&self, _: &mut ()) -> io::Result<()> {
        if self.unknown.load(Ordering::SeqCst) {
            Err(io::Error::other("original stream still unknown"))
        } else {
            self.submitted.store(false, Ordering::SeqCst);
            Ok(())
        }
    }
    fn readback(
        &self,
        _: &mut (),
        _: &Self::Buffer,
        _: CopyRegion,
        _: HostTransferLayout,
    ) -> io::Result<Vec<u8>> {
        panic!("managed lifecycle must not allocate a raw readback payload")
    }
    fn readback_managed_into(
        &self,
        _: &mut (),
        _: &Self::Buffer,
        region: CopyRegion,
        layout: HostTransferLayout,
        destination: &mut ManagedHostBuffer,
    ) -> io::Result<bool> {
        let start = region.destination_offset_bytes() as usize;
        let end = start + layout.byte_len().unwrap() as usize;
        destination.as_mut()[start..end].fill(0);
        Ok(true)
    }
    fn describe_error(&self, error: &io::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("component_custody_fixture", error.to_string(), false)
    }
}

fn phase() -> (HostMemoryAccount, HostMemoryScope) {
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            512,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128,
                transfer_staging_bytes: 128,
                output_bytes: 128,
                weight_backing_bytes: 128,
            },
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let phase = host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 128),
            (HostMemoryClass::TransferStaging, 128),
            (HostMemoryClass::Output, 128),
            (HostMemoryClass::WeightBacking, 128),
        ])
        .unwrap();
    (host, phase)
}

#[test]
fn component_retained_preflight_preserves_typed_error_and_returns_original_phase() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: true,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    let (parent, phase) = phase();
    let error = compiled
        .initialize_retained(
            runtime.clone(),
            phase.account(),
            ComponentWeightSource::None,
            init_config(),
            phase,
        )
        .err()
        .unwrap();
    assert!(matches!(
        error.error(),
        ComponentInitializationError::Contract(_)
    ));
    assert_eq!(runtime.allocation_calls.load(Ordering::SeqCst), 0);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 512);
    let (_, original) = error.try_into_preflight().unwrap();
    original.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn component_retained_actual_close_hands_back_scope_after_referenced_retry() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: false,
        persistent: true,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    let (parent, phase) = phase();
    let session = compiled
        .initialize_retained(
            runtime.clone(),
            phase.account(),
            ComponentWeightSource::None,
            init_config(),
            phase,
        )
        .unwrap();
    assert!(runtime.allocation_calls.load(Ordering::SeqCst) > 0);
    let alias = session.session().resources.as_ref().unwrap().clone();
    let expected = alias.trusted_runtime_binding().unwrap().evidence();
    let pending = session.close().err().unwrap();
    assert!(pending.resources().is_some());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 512);
    drop(alias);
    let (phase, receipt) = pending.retry().unwrap();
    assert_eq!(receipt.evidence(), &expected);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        512,
        "Closed returns the actual grant, not a new account"
    );
    phase.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

struct SourceFailure {
    panic: bool,
}
impl WeightComponentSource for SourceFailure {
    fn component<'a>(
        &'a self,
        _: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        assert!(!self.panic, "source panic after native allocation");
        Err(invalid("source conversion refused"))
    }
}

#[test]
fn component_retained_static_failure_uses_original_checked_retirement() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: true,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    let (parent, phase) = phase();
    let mut error = compiled
        .initialize_retained(
            runtime.clone(),
            phase.account(),
            ComponentWeightSource::Stored(&SourceFailure { panic: false }),
            init_config(),
            phase,
        )
        .err()
        .unwrap();
    assert!(
        matches!(error.error(), ComponentInitializationError::Static(failure) if !failure.is_retired())
    );
    assert!(runtime.allocation_calls.load(Ordering::SeqCst) > 0);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 512);
    assert!(error.retire_static_in_place().unwrap());
    let (_, phase) = error.try_into_retired().unwrap();
    phase.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn component_retained_pre_native_panic_drops_phase_without_fabricating_close() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: false,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    runtime.descriptor_panic.store(true, Ordering::SeqCst);
    let (parent, phase) = phase();
    let panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = compiled.initialize_retained(
            runtime,
            phase.account(),
            ComponentWeightSource::None,
            init_config(),
            phase,
        );
    }));
    assert!(panic.is_err());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn component_retained_native_source_panic_quarantines_original_scope() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: true,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    let (parent, phase) = phase();
    let panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = compiled.initialize_retained(
            runtime.clone(),
            phase.account(),
            ComponentWeightSource::Stored(&SourceFailure { panic: true }),
            init_config(),
            phase,
        );
    }));
    assert!(panic.is_err());
    assert!(runtime.allocation_calls.load(Ordering::SeqCst) > 0);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        512,
        "no original Closed/retired authority exists after this raw panic"
    );
}

#[test]
fn component_retained_constructor_panic_after_binding_does_not_refund_phase() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: false,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    runtime.managed_panic.store(true, Ordering::SeqCst);
    let (parent, phase) = phase();
    let panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = compiled.initialize_retained(
            runtime,
            phase.account(),
            ComponentWeightSource::None,
            init_config(),
            phase,
        );
    }));
    assert!(panic.is_err());
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        512,
        "root abandonment is not an actual close receipt"
    );
}

#[test]
fn component_retained_unknown_execution_keeps_phase_until_original_fence_and_close() {
    let compiled = compile_for::<Runtime>(&Config {
        stored: false,
        persistent: false,
    });
    let runtime = Runtime::new(compiled.executable().execution_plan());
    let (parent, phase) = phase();
    let host = phase.account();
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            init_config(),
            phase,
        )
        .unwrap();
    let input = ManagedHostTensor::new(
        f::id("value.input"),
        ResolvedTensorSpec::new(vec![4], ElementType::F32, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        host.allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap()
            .freeze(),
    )
    .unwrap();
    runtime.unknown.store(true, Ordering::SeqCst);
    let error = session
        .session_mut()
        .execute(
            f::id("run.custody.unknown"),
            f::id("request.custody.unknown"),
            vec![input],
        )
        .err()
        .unwrap();
    let TensorExecutionError::NonTerminal {
        completion,
        mut retirement,
        reaper,
        ..
    } = error
    else {
        panic!("original indeterminate fence must retain its actual completion owner: {error:?}");
    };
    let pending = session.close().err().unwrap();
    assert!(pending.resources().is_some());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 512);
    runtime.unknown.store(false, Ordering::SeqCst);
    assert!(matches!(
        completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    retirement.try_abort().unwrap();
    drop((completion, retirement, reaper));
    let (phase, _) = pending.retry().unwrap();
    phase.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
