//! Real CUDA static uploads, using the existing registered BF16 Qwen fixture.
//! No inference or throughput claim: every installed component is read back
//! through its original Core buffer before the real release acknowledgement.
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_and_dense_f16_causal_runtime_config, cuda_vnext_dense_f16_causal_contracts,
        cuda_vnext_dense_f16_causal_providers,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorAndCausalRuntimeConfig, CudaTensorRuntimeConfig},
    CudaDenseF16CausalProfile,
};
use ferrum_runtime::*;
use std::{collections::BTreeMap, sync::Mutex};

const DEVICE_BYTES: u64 = 2 << 30;
const CHUNK_BYTES: u64 = 64;

fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> T {
    s.to_owned().try_into().unwrap()
}

fn composition() -> (
    RuntimeComposition<CudaDeviceRuntime>,
    Arc<CudaDeviceRuntime>,
) {
    // Conservative declared smoke allowances, not measured opaque footprints.
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_and_causal_eager(CudaTensorAndCausalRuntimeConfig {
            tensor: CudaTensorRuntimeConfig {
                device: cuda_tensor_and_dense_f16_causal_runtime_config(0, id("device.cuda.0"))
                    .unwrap(),
                usable_memory_bytes: DEVICE_BYTES,
                bootstrap_memory_bytes: 1 << 29,
                stream_memory_bytes: 64 << 20,
                maximum_streams: 1,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 64,
                opaque_memory_profile: "test.static-reuse.cuda.v1".into(),
            },
            causal: CudaDenseF16CausalProfile::new(1 << 28, "test.static-reuse.causal.v1").unwrap(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        cuda_vnext_dense_f16_causal_contracts().unwrap(),
        cuda_vnext_dense_f16_causal_providers(&runtime).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let device = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.reusable-source"),
        ContractVersion::new(1, 0),
        device.runtime_implementation_fingerprint.clone(),
        device.id.clone(),
        device.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(device.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    (
        RuntimeComposition::new(runtime.clone(), registry, materializers, catalog),
        runtime,
    )
}

fn compile(
    composition: &RuntimeComposition<CudaDeviceRuntime>,
    prepared: &PreparedProductionModel,
) -> ExecutionPlan {
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CUDA(0);
    engine.backend.enable_reusable_execution = false;
    engine.scheduler.max_running_requests = 2;
    engine.batching.max_num_batched_tokens = 8;
    engine.runtime.max_model_len = Some(128);
    let info = prepared.model_info(engine.model.model_id.clone(), engine.backend.device.clone());
    let config =
        ProgramExecutorConfig::from_engine_config(&engine, &info, composition.runtime()).unwrap();
    let compiled = composition
        .compile_model(
            prepared.execution_input().unwrap(),
            info,
            &engine,
            config,
            WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
        )
        .unwrap();
    compiled.compilation().executable().execution_plan().clone()
}

/// Distinguish every source, its element order, and the smaller tail. Reusing
/// a slot too soon must not accidentally pass because all matrices are equal.
fn distinct_fixture() -> Fixture {
    let fixture = Fixture::new();
    let original = std::fs::read(&fixture.selected).unwrap();
    let archive = safetensors::SafeTensors::deserialize(&original).unwrap();
    let mut tensors = archive.tensors();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let owned = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, tensor))| {
            let bytes = (0..tensor.shape().iter().product::<usize>())
                .flat_map(|n| {
                    let value = (i + 1) as f32 / 128. + (n % 5) as f32 / 1024.;
                    bf16::from_f32(value).to_bits().to_le_bytes()
                })
                .collect::<Vec<_>>();
            (name.clone(), tensor.shape().to_vec(), bytes)
        })
        .collect::<Vec<_>>();
    let views = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &fixture.selected).unwrap();
    fixture
}

struct Probe<'a, 'source> {
    source: &'a crate::vnext::ManagedProductionWeightSource<'source>,
    host: &'a HostMemoryAccount,
    peak: usize,
    expected: &'a BTreeMap<WeightId, Vec<u8>>,
    observations: Mutex<Vec<(WeightId, usize, usize)>>,
}
impl WeightComponentSource for Probe<'_, '_> {
    fn validate_static_initialization(
        &self,
        context: &StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        assert_eq!(
            context.source_retention(),
            StaticWeightSourceRetention::FencedCopy
        );
        assert!(context.matches_host_account(self.host));
        self.source.validate_static_initialization(context)
    }

    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let payload = self.source.component(c)?;
        assert_eq!(payload.bytes(), self.expected[&c.id].as_slice());
        assert_eq!(payload.bytes().len() as u64, c.physical_bytes().unwrap());
        assert_eq!(
            payload
                .retained_host_memory()
                .unwrap()
                .managed_memory_class(),
            Some(HostMemoryClass::WeightBacking)
        );
        assert_eq!(
            self.host
                .usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::WeightBacking),
            self.peak
        );
        self.observations.lock().unwrap().push((
            c.id.clone(),
            payload.bytes().as_ptr() as usize,
            payload.bytes().len(),
        ));
        Ok(payload)
    }
}

#[derive(Clone)]
struct ExpectedPlacement {
    component: WeightId,
    resource: ResourceId,
    offset: u64,
    element: ElementType,
    bytes: Vec<u8>,
}

fn placements(
    plan: &ExecutionPlan,
    expected: &BTreeMap<WeightId, Vec<u8>>,
) -> Vec<ExpectedPlacement> {
    let mut placed = BTreeMap::new();
    for node in plan.payload().nodes() {
        for value in node
            .values()
            .iter()
            .filter(|v| v.usage() == BufferUsage::Weights)
        {
            for c in value.storage().components() {
                let component = c.component_id().unwrap();
                let bytes = expected[component].clone();
                assert_eq!(bytes.len() as u64, c.length_bytes());
                let geometry = (c.resource_id().clone(), c.offset_bytes(), c.element_type());
                if let Some((old, _)) = placed.insert(component.clone(), (geometry.clone(), bytes))
                {
                    assert_eq!(old, geometry);
                }
            }
        }
    }
    placed
        .into_iter()
        .map(
            |(component, ((resource, offset, element), bytes))| ExpectedPlacement {
                component,
                resource,
                offset,
                element,
                bytes,
            },
        )
        .collect()
}

/// Delegate all allocation/ownership to the production driver. The release
/// callback borrows the original still-live Core buffer, so no public pointer
/// accessor, duplicate allocation, fake state-copy ticket or CUDA alias is used.
struct ReadbackDriver {
    inner: RuntimeResourceDriver<CudaDeviceRuntime>,
    host: HostMemoryAccount,
    expected: Vec<ExpectedPlacement>,
    verified: Arc<Mutex<Vec<WeightId>>>,
}
impl ResourceTransactionDriver for ReadbackDriver {
    type Runtime = CudaDeviceRuntime;
    type Buffer = <CudaDeviceRuntime as DeviceRuntime>::Buffer;
    fn runtime(&self) -> &Arc<Self::Runtime> {
        self.inner.runtime()
    }
    fn device_id(&self) -> &DeviceId {
        self.inner.device_id()
    }
    fn device_runtime_implementation_fingerprint(&self) -> &str {
        self.inner.device_runtime_implementation_fingerprint()
    }
    fn device_capacity_bytes(&self) -> u64 {
        self.inner.device_capacity_bytes()
    }
    fn reserve_resource(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.reserve_resource(c, r)
    }
    fn commit_resource<'c>(
        &mut self,
        c: &'c ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'c>, ResourceDriverFailure> {
        self.inner.commit_resource(c, r)
    }
    fn compensate_reserve_resource(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.compensate_reserve_resource(c, r)
    }
    fn compensate_commit_resource(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
        b: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.compensate_commit_resource(c, r, b)
    }
    fn rollback_resource(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.rollback_resource(c, r)
    }
    fn release_resource(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
        b: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        let mut stream = self.inner.runtime().create_stream().unwrap();
        for expected in self
            .expected
            .iter()
            .filter(|p| &p.resource == r.resource_id())
        {
            let mut output = self
                .host
                .allocate_zeroed(HostMemoryClass::Output, expected.bytes.len())
                .unwrap();
            assert!(self
                .inner
                .runtime()
                .readback_managed_into(
                    &mut stream,
                    b,
                    CopyRegion::new(expected.offset, 0, expected.bytes.len() as u64).unwrap(),
                    HostTransferLayout::new(
                        expected.element,
                        expected.bytes.len() as u64 / expected.element.size_bytes()
                    )
                    .unwrap(),
                    &mut output,
                )
                .unwrap());
            assert_eq!(
                output.as_mut(),
                expected.bytes.as_slice(),
                "installed {} was overwritten before its copy completed",
                expected.component
            );
            self.verified
                .lock()
                .unwrap()
                .push(expected.component.clone());
        }
        drop(stream);
        self.inner.release_resource(c, r, b)
    }
    fn reconcile_commit_outcome(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        r: &ResourceReservation,
        actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.reconcile_commit_outcome(c, r, actual)
    }
    fn quarantine_transaction(
        &mut self,
        c: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        self.inner.quarantine_transaction(c, ownership)
    }
    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        self.inner.abandon_transaction(ownership)
    }
}

fn committed<D: ResourceTransactionDriver<Runtime = CudaDeviceRuntime>>(
    plan: &ExecutionPlan,
    driver: D,
) -> ResourceTransaction<D, TransactionCommitted> {
    let StaticProvisioning::Required(permit) = plan
        .provision_static(driver.runtime().clone(), id("request.cuda.reusable-source"))
        .unwrap()
        .into_provisioning()
    else {
        panic!("real Stored fixture needs static allocation")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.cuda.reusable-source"),
        id("transaction.cuda.reusable-source"),
    );
    ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap()
        .commit()
        .unwrap_or_else(|error| match error {
            ResourceCommitTransitionError::Recoverable(f) => {
                panic!("static commit: {:?}", f.failure())
            }
            ResourceCommitTransitionError::Poisoned(f) => {
                panic!("indeterminate static commit: {:?}", f.failure())
            }
        })
}

#[test]
#[ignore = "requires exclusive CUDA and pinned native operator-set lock; real multi-component direct-source static copies"]
fn reusable_cuda_static_source_copies_distinct_components_and_retires_original_owners() {
    let fixture = distinct_fixture();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let (composition, runtime) = composition();
    let plan = compile(&composition, &prepared);
    assert!(runtime.supports_managed_static_weight_sources());
    assert!(runtime.begin_managed_static_weight_import().is_none());
    let parent = host(1 << 18);
    let expected = {
        let fresh = mapped.weight_source(&parent);
        prepared
            .family()
            .stored_weight_schema()
            .unwrap()
            .components
            .iter()
            .map(|c| (c.id.clone(), fresh.component(c).unwrap().bytes().to_vec()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    let placed = placements(&plan, &expected);
    assert!(placed.len() > 1);
    assert!(placed.windows(2).any(|p| p[0].bytes != p[1].bytes));
    assert!(placed.iter().any(|p| p.bytes.len() > CHUNK_BYTES as usize));
    let peak = mapped
        .weight_source(&parent)
        .maximum_group_backing_bytes(&plan)
        .unwrap();
    let output_peak = placed
        .iter()
        .map(|p| parent.backing_bytes(p.bytes.len()).unwrap())
        .max()
        .unwrap();
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, peak),
            (
                HostMemoryClass::TransferStaging,
                parent.backing_bytes(CHUNK_BYTES as usize).unwrap(),
            ),
            (HostMemoryClass::Output, output_peak),
        ])
        .unwrap();
    let account = scope.account();
    let source = mapped.weight_source(&account);
    source.prepare_reusable_static_backing(&plan).unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    let probe = Probe {
        source: &source,
        host: &account,
        peak,
        expected: &expected,
        observations: Mutex::new(Vec::new()),
    };
    let verified = Arc::new(Mutex::new(Vec::new()));
    let initialized = committed(
        &plan,
        ReadbackDriver {
            inner: RuntimeResourceDriver::new(runtime.clone()).unwrap(),
            host: account.clone(),
            expected: placed.clone(),
            verified: verified.clone(),
        },
    )
    .initialize_static_managed(
        prepared.family(),
        &plan,
        &probe,
        &account,
        StaticInitializationPolicy::new(CHUNK_BYTES, 2).unwrap(),
    )
    .unwrap();
    let total: u64 = placed.iter().map(|p| p.bytes.len() as u64).sum();
    assert_eq!(initialized.receipt().uploaded_bytes(), total);
    assert_eq!(
        initialized.receipt().uploaded_component_count(),
        placed.len()
    );
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), total);
    assert_eq!(initialized.receipt().imported_component_count(), 0);
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 0);
    assert!(initialized.receipt().upload_command_count() > placed.len());
    // Every converted identity group must drain its original source owners
    // before the next group can check out this same slot.
    assert!(initialized.receipt().submission_batch_count() >= placed.len());
    let observations = probe.observations.lock().unwrap();
    assert_eq!(observations.len(), placed.len());
    assert!(
        observations.iter().all(|v| v.1 == observations[0].1),
        "all logical payloads retain the same full allocation"
    );
    assert!(
        observations.windows(2).any(|v| v[0].2 != v[1].2),
        "real unequal payload extents exercise the slot tail"
    );
    let address = observations[0].1;
    drop(observations);
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        peak
    );
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::TransferStaging),
        0
    );

    // The initializer's original group fences have all completed and dropped.
    // A successful thaw now proves no upload command still owns the slot.
    let specs = &prepared.family().stored_weight_schema().unwrap().components;
    let payload = source.component(&specs[0]).unwrap();
    assert_eq!(payload.bytes().as_ptr() as usize, address);
    let retained = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    assert!(source.component(&specs[1]).is_err());
    drop(retained);
    let payload = source.component(&specs[1]).unwrap();
    assert_eq!(payload.bytes().as_ptr() as usize, address);
    let retained = payload.retained_host_memory().unwrap().clone();
    drop(payload);

    // Deliberately overwrite the source slot after upload, before D2H. Every
    // earlier installed component must still match its fresh-source bytes.
    let resources = initialized.into_plan_runtime().unwrap();
    let weak = Arc::downgrade(&resources);
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert!(receipt.released_static_resources() > 0)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("unexpected plan owners: {strong_count}")
        }
        Err(failure) => panic!("original CUDA retirement failed: {:?}", failure.failure()),
    }
    assert!(weak.upgrade().is_none());
    let mut actual = verified.lock().unwrap().clone();
    actual.sort();
    assert_eq!(
        actual,
        placed
            .iter()
            .map(|p| p.component.clone())
            .collect::<Vec<_>>()
    );
    drop(probe);
    drop(source);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), peak);
    assert_eq!(parent.usage().unwrap().committed_bytes(), peak);
    drop(retained);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);

    // Original physical claims are reusable after acknowledged plan close.
    // Runtime/bootstrap/provider allowances may still be live; no VRAM-zero
    // claim is made for those independent owners or the native allocation pool.
    let next = committed(&plan, RuntimeResourceDriver::new(runtime).unwrap());
    drop(
        next.release()
            .unwrap_or_else(|failure| panic!("new transaction release: {:?}", failure.failure())),
    );
}
