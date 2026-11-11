//! Actual static installation, not causal token I/O qualification. Reuse the
//! existing Stored causal-metadata family and physical plan; CPU commands copy
//! the converted weights into the original admitted native buffers.
use super::*;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Mutex;

mod phase_custody;

#[path = "../../../tests/capacity_boundary/fixture.rs"]
#[allow(dead_code)]
mod fixture;

struct Storage {
    bytes: Mutex<Vec<u8>>,
    live: Arc<AtomicUsize>,
}
impl Drop for Storage {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Buffer {
    descriptor: BufferDescriptor,
    storage: Arc<Storage>,
}
enum UploadBytes {
    Managed(ManagedHostBytes),
    Legacy(Vec<u8>),
}
enum Command {
    Upload(UploadBytes, Arc<Storage>, usize),
    Zero(Arc<Storage>, usize, usize),
}
type Pending = Arc<Mutex<Option<Vec<Command>>>>;
#[derive(Default)]
struct Stream {
    pending: Vec<Pending>,
}
struct Fence {
    pending: Pending,
    has_upload: bool,
}
struct Upload {
    storage: Weak<Storage>,
    offset: usize,
    bytes: Vec<u8>,
}
#[derive(Default)]
struct Trace {
    uploads: Vec<Upload>,
    managed_uploads: usize,
    raw_uploads: usize,
    submissions: usize,
    simultaneous_backing_and_staging: bool,
}
struct Runtime {
    descriptor: DeviceDescriptor,
    host: HostMemoryAccount,
    live: Arc<AtomicUsize>,
    trace: Mutex<Trace>,
    unknown_upload_fence: AtomicBool,
    allow_cleanup: AtomicBool,
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
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> io::Result<Buffer> {
        let request = permit.into_request();
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(Buffer {
            descriptor: BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
            storage: Arc::new(Storage {
                bytes: Mutex::new(vec![0xcd; request.size_bytes() as usize]),
                live: self.live.clone(),
            }),
        })
    }
    fn buffer_descriptor(&self, buffer: &Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }
    fn create_stream(&self) -> io::Result<Stream> {
        Ok(Stream::default())
    }
    fn stream_state(&self, stream: &Stream) -> StreamState {
        if stream.pending.iter().any(|p| p.lock().unwrap().is_some()) {
            StreamState::Submitted
        } else {
            StreamState::Ready
        }
    }
    fn encode_copy(&self, _: &Buffer, _: &Buffer, _: CopyRegion) -> io::Result<Command> {
        Err(io::Error::other(
            "static fixture does not perform device copies",
        ))
    }
    fn encode_upload(
        &self,
        source: &[u8],
        _: HostTransferLayout,
        destination: &Buffer,
        offset: u64,
    ) -> io::Result<Command> {
        self.trace.lock().unwrap().raw_uploads += 1;
        Ok(Command::Upload(
            UploadBytes::Legacy(source.to_vec()),
            destination.storage.clone(),
            offset as usize,
        ))
    }
    fn supports_managed_tensor_io(&self) -> bool {
        true
    }
    fn encode_managed_upload(
        &self,
        source: ManagedHostBytes,
        _: HostTransferLayout,
        destination: &Buffer,
        offset: u64,
    ) -> io::Result<Option<Command>> {
        assert!(self.host.owns(&source));
        assert_eq!(source.class(), HostMemoryClass::TransferStaging);
        let usage = self.host.usage().unwrap();
        let mut trace = self.trace.lock().unwrap();
        trace.managed_uploads += 1;
        trace.simultaneous_backing_and_staging |=
            usage.class_allocated_bytes(HostMemoryClass::WeightBacking) == 16
                && usage.class_allocated_bytes(HostMemoryClass::TransferStaging) == 8;
        Ok(Some(Command::Upload(
            UploadBytes::Managed(source),
            destination.storage.clone(),
            offset as usize,
        )))
    }
    fn encode_zero(&self, buffer: &Buffer, offset: u64, length: u64) -> io::Result<Command> {
        Ok(Command::Zero(
            buffer.storage.clone(),
            offset as usize,
            length as usize,
        ))
    }
    fn submit(
        &self,
        stream: &mut Stream,
        batch: DeviceCommandBatch<Command>,
    ) -> std::result::Result<Fence, DefinitelyNotSubmitted<io::Error>> {
        let commands = batch.into_commands();
        let mut trace = self.trace.lock().unwrap();
        trace.submissions += 1;
        let mut has_upload = false;
        for command in &commands {
            match command {
                Command::Zero(storage, offset, length) => {
                    storage.bytes.lock().unwrap()[*offset..offset + length].fill(0);
                }
                Command::Upload(source, storage, offset) => {
                    let bytes = match source {
                        UploadBytes::Managed(owner) => owner.bytes(),
                        UploadBytes::Legacy(bytes) => bytes,
                    };
                    storage.bytes.lock().unwrap()[*offset..offset + bytes.len()]
                        .copy_from_slice(bytes);
                    trace.uploads.push(Upload {
                        storage: Arc::downgrade(storage),
                        offset: *offset,
                        bytes: bytes.to_vec(),
                    });
                    has_upload = true;
                }
            }
        }
        let pending = Arc::new(Mutex::new(Some(commands)));
        stream.pending.push(pending.clone());
        Ok(Fence {
            pending,
            has_upload,
        })
    }
    fn query_fence(&self, _: &Fence) -> FenceQuery<io::Error> {
        FenceQuery::Pending
    }
    fn wait_fence(
        &self,
        fence: &Fence,
    ) -> std::result::Result<DeviceTerminalReceipt<io::Error>, FenceIndeterminate<io::Error>> {
        if fence.has_upload && self.unknown_upload_fence.load(Ordering::SeqCst) {
            return Err(FenceIndeterminate::new(io::Error::other(
                "upload completion is unknown",
            )));
        }
        fence.pending.lock().unwrap().take();
        Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }
    fn synchronize(&self, stream: &mut Stream) -> io::Result<()> {
        if !self.allow_cleanup.load(Ordering::SeqCst) {
            return Err(io::Error::other("original upload remains in flight"));
        }
        for pending in stream.pending.drain(..) {
            pending.lock().unwrap().take();
        }
        Ok(())
    }
    fn readback(
        &self,
        _: &mut Stream,
        _: &Buffer,
        _: CopyRegion,
        _: HostTransferLayout,
    ) -> io::Result<Vec<u8>> {
        Err(io::Error::other(
            "static fixture does not perform token readback",
        ))
    }
    fn describe_error(
        &self,
        error: &io::Error,
    ) -> std::result::Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("static_host_fixture", error.to_string(), false)
    }
}

struct ConvertedSource {
    host: HostMemoryAccount,
    attempts: AtomicUsize,
}
impl WeightComponentSource for ConvertedSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        assert_eq!(component.dimensions, [4]);
        assert_eq!(component.physical_element_type(), ElementType::F32);
        let mut bytes = self
            .host
            .allocate_zeroed(HostMemoryClass::WeightBacking, 16)
            .map_err(|error| VNextError::InvalidExecutionPlan {
                reason: error.to_string(),
            })?;
        for (slot, value) in bytes
            .as_mut()
            .chunks_exact_mut(4)
            .zip([1.0, -0.5, 3.0, 0.125])
        {
            slot.copy_from_slice(&half::f16::from_f32(value).to_f32().to_le_bytes());
        }
        WeightComponentPayload::from_managed_host(
            component,
            component.external_names.clone(),
            vec!["model.safetensors".into()],
            component.dimensions.clone(),
            ElementType::F32,
            bytes.freeze(),
        )
    }
}

fn account(backing: usize, staging: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            backing + staging,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: staging,
                output_bytes: 0,
                weight_backing_bytes: backing,
            },
        )
        .unwrap(),
    )
}
fn setup(host: &HostMemoryAccount) -> (PreparedModelFamily, ExecutionPlan, Arc<Runtime>) {
    let family = TypedFamilyRegistration::new(fixture::TestFamily)
        .prepare_with_profile(
            &serde_json::json!({"width": 4}),
            &NumericalProfileId::new("fixture.f32").unwrap(),
        )
        .unwrap();
    family.metadata().causal_language().unwrap();
    let catalog = fixture::catalog();
    let (plan, _) = fixture::plan(&catalog, false, 1);
    let runtime = Arc::new(Runtime {
        descriptor: catalog.device().clone(),
        host: host.clone(),
        live: Arc::new(AtomicUsize::new(0)),
        trace: Mutex::new(Trace::default()),
        unknown_upload_fence: AtomicBool::new(false),
        allow_cleanup: AtomicBool::new(true),
    });
    (family, plan, runtime)
}
fn committed(
    plan: &ExecutionPlan,
    runtime: &Arc<Runtime>,
) -> ResourceTransaction<RuntimeResourceDriver<Runtime>, TransactionCommitted> {
    let provisioning = plan
        .provision_static(runtime.clone(), fixture::id("request.static-host"))
        .unwrap();
    let StaticProvisioning::Required(permit) = provisioning.into_provisioning() else {
        panic!("Stored fixture requires an actual static transaction")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        fixture::id("run.static-host"),
        fixture::id("transaction.static-host"),
    );
    ResourceTransaction::begin(
        RuntimeResourceDriver::new(runtime.clone()).unwrap(),
        identity,
        permit,
    )
    .unwrap()
    .reserve()
    .unwrap()
    .commit()
    .unwrap_or_else(|error| match error {
        ResourceCommitTransitionError::Recoverable(failure) => {
            panic!("actual static commit failed: {:?}", failure.failure())
        }
        ResourceCommitTransitionError::Poisoned(failure) => {
            panic!(
                "actual static commit indeterminate: {:?}",
                failure.failure()
            )
        }
    })
}
fn source(host: HostMemoryAccount) -> ConvertedSource {
    ConvertedSource {
        host,
        attempts: AtomicUsize::new(0),
    }
}
fn policy() -> StaticInitializationPolicy {
    StaticInitializationPolicy::new(8, 1).unwrap()
}

#[test]
fn managed_static_install_converts_and_uploads_with_one_original_scope() {
    let host = account(16, 8);
    let scope = host
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, 16),
            (HostMemoryClass::TransferStaging, 8),
        ])
        .unwrap();
    let scoped = scope.account();
    let source = source(scoped.clone());
    let (family, plan, runtime) = setup(&host);
    let initialized = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        Some(&scoped),
        policy(),
    )
    .unwrap();
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    assert_eq!(source.attempts.load(Ordering::SeqCst), 1);
    let root = initialized.into_plan_runtime().unwrap();
    {
        let trace = runtime.trace.lock().unwrap();
        assert_eq!(trace.managed_uploads, 2);
        assert_eq!(trace.raw_uploads, 0);
        assert!(trace.simultaneous_backing_and_staging);
        let expected = [1.0_f32, -0.5, 3.0, 0.125]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            trace
                .uploads
                .iter()
                .flat_map(|u| u.bytes.iter().copied())
                .collect::<Vec<_>>(),
            expected
        );
        for upload in &trace.uploads {
            let storage = upload.storage.upgrade().unwrap();
            assert_eq!(
                &storage.bytes.lock().unwrap()[upload.offset..upload.offset + upload.bytes.len()],
                upload.bytes
            );
        }
    }
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
        Ok(_) => panic!("static owner unexpectedly referenced"),
        Err(failure) => panic!("static close failed: {:?}", failure.failure()),
    }
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    drop((source, scoped, scope));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn managed_static_install_rejects_real_staging_and_conversion_shortfalls() {
    for (backing, staging, attempts) in [(16, 7, 0), (15, 8, 1)] {
        let host = account(backing, staging);
        let source = source(host.clone());
        let (family, plan, runtime) = setup(&host);
        let failure = initialize_static_with_host(
            committed(&plan, &runtime),
            &family,
            &plan,
            &source,
            Some(&host),
            policy(),
        )
        .err()
        .expect("one byte below actual class demand must refuse initialization");
        assert!(!failure.is_indeterminate());
        assert!(
            runtime.live.load(Ordering::SeqCst) > 0,
            "failure retains its original admitted transaction"
        );
        assert_eq!(source.attempts.load(Ordering::SeqCst), attempts);
        assert_eq!(runtime.trace.lock().unwrap().managed_uploads, 0);
        let error =
            shutdown::initialization_error(failure, FerrumError::device("static Host refused"));
        assert!(
            error.cleanup().is_none(),
            "quiescent cleanup completes synchronously"
        );
        assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn legacy_static_install_keeps_its_explicit_unmanaged_transfer_behavior() {
    let host = account(16, 0);
    let source = source(host.clone());
    let (family, plan, runtime) = setup(&host);
    let initialized = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        None,
        policy(),
    )
    .unwrap();
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    let mut failure = initialized.abort_activation(VNextError::InvalidExecutionPlan {
        reason: "test closes the original initialized transaction".into(),
    });
    failure.retire_in_place().unwrap();
    let trace = runtime.trace.lock().unwrap();
    assert_eq!(trace.raw_uploads, 2);
    assert_eq!(trace.managed_uploads, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unknown_static_upload_retains_original_host_and_native_until_cleanup_ack() {
    let host = account(16, 8);
    let source = source(host.clone());
    let (family, plan, runtime) = setup(&host);
    runtime.unknown_upload_fence.store(true, Ordering::SeqCst);
    runtime.allow_cleanup.store(false, Ordering::SeqCst);
    let failure = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        Some(&host),
        policy(),
    )
    .err()
    .expect("actual upload fence becomes indeterminate");
    assert!(failure.is_indeterminate());
    let error =
        shutdown::initialization_error(failure, FerrumError::device("unknown static fence"));
    let mut cleanup = error
        .cleanup()
        .expect("original cleanup custody is retained")
        .clone();
    let report = tokio::time::timeout(Duration::from_secs(5), cleanup.wait())
        .await
        .unwrap();
    assert_eq!(report.phase(), crate::ExecutorShutdownPhase::Retained);
    assert!(runtime.live.load(Ordering::SeqCst) > 0);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::TransferStaging),
        8
    );
    let submissions = runtime.trace.lock().unwrap().submissions;
    runtime.allow_cleanup.store(true, Ordering::SeqCst);
    cleanup.retry();
    let report = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let report = cleanup.changed().await;
            if report.phase() == crate::ExecutorShutdownPhase::Retired {
                break report;
            }
        }
    })
    .await
    .unwrap();
    assert!(
        report.error().is_some(),
        "safe retirement does not erase the original failure"
    );
    assert_eq!(
        runtime.trace.lock().unwrap().submissions,
        submissions,
        "cleanup never replays uploads"
    );
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn shared_composition_retains_original_runtime_registry_and_materializers() {
    let catalog = fixture::catalog();
    let (plan, _) = fixture::plan(&catalog, false, 1);
    let runtime = fixture::runtime(&plan);
    let registry = Arc::new(fixture::operation_registry(&catalog));
    let materializers = Arc::new(WeightMaterializerRegistry::identity_only().unwrap());
    let composition = RuntimeComposition::from_shared(
        runtime.clone(),
        registry.clone(),
        materializers.clone(),
        catalog.clone(),
    );
    assert!(Arc::ptr_eq(&composition.runtime, &runtime));
    assert!(Arc::ptr_eq(&composition.registry, &registry));
    assert!(Arc::ptr_eq(
        &composition.weight_materializers,
        &materializers
    ));
    assert_eq!(
        composition.runtime().descriptor(),
        composition.catalog().device()
    );
}
