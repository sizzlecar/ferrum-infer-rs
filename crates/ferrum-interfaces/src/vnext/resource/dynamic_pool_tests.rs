use super::*;
use crate::vnext::{
    CapacityAvailabilitySource, CopyRegion, DeferredAction, DefinitelyNotSubmitted,
    DeviceCapacityPressureScope, DeviceClass, DeviceCommandBatch, DeviceErrorReport,
    DeviceTerminal, DeviceTerminalReceipt, FenceIndeterminate, FenceQuery, HostTransferLayout,
    TrustedActiveSequenceBinding,
};
use serde_json::{json, Value};
use std::error::Error;

static NEXT_TEST_DEVICE: AtomicU64 = AtomicU64::new(1);
const DYNAMIC_POOL_CONCURRENT_WORKERS: usize = 1;
const MAX_DYNAMIC_POOL_TEST_WORKERS: usize = 2;
const _: () = assert!(
    DYNAMIC_POOL_CONCURRENT_WORKERS == 1
        && DYNAMIC_POOL_CONCURRENT_WORKERS <= MAX_DYNAMIC_POOL_TEST_WORKERS
);

#[derive(Debug)]
struct TestRuntimeError;

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("injected dynamic-pool allocation failure")
    }
}

impl Error for TestRuntimeError {}

struct TestBuffer {
    descriptor: BufferDescriptor,
    backend_alive: Weak<AtomicBool>,
    dropped_after_backend: Arc<AtomicBool>,
}

impl Drop for TestBuffer {
    fn drop(&mut self) {
        if self
            .backend_alive
            .upgrade()
            .is_none_or(|alive| !alive.load(Ordering::Acquire))
        {
            self.dropped_after_backend.store(true, Ordering::Release);
        }
    }
}

struct TestStream {
    drops: Arc<AtomicU64>,
}

impl Drop for TestStream {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
    }
}

struct TestRuntime {
    descriptor: DeviceDescriptor,
    allocate_calls: AtomicU64,
    fail_on_call: AtomicU64,
    panic_on_call: AtomicU64,
    mismatch_on_call: AtomicU64,
    backend_alive: Arc<AtomicBool>,
    dropped_after_backend: Arc<AtomicBool>,
    synchronize_calls: AtomicU64,
    synchronize_failures: AtomicU64,
    stream_drops: Arc<AtomicU64>,
    allocation_rendezvous: Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    close_returned_probe: Mutex<Option<Arc<AtomicBool>>>,
    observed_close_return_during_allocation: AtomicBool,
}

impl Drop for TestRuntime {
    fn drop(&mut self) {
        self.backend_alive.store(false, Ordering::Release);
    }
}

impl TestRuntime {
    fn new(
        device_id: DeviceId,
        total_memory_bytes: u64,
        profiles: BTreeSet<DynamicStorageProfile>,
    ) -> Self {
        Self {
            descriptor: DeviceDescriptor {
                id: device_id,
                class: DeviceClass::Reference,
                ordinal: 0,
                total_memory_bytes,
                runtime_implementation_fingerprint: "d".repeat(64),
                capabilities: BTreeSet::new(),
                dynamic_storage_profiles: profiles,
            },
            allocate_calls: AtomicU64::new(0),
            fail_on_call: AtomicU64::new(0),
            panic_on_call: AtomicU64::new(0),
            mismatch_on_call: AtomicU64::new(0),
            backend_alive: Arc::new(AtomicBool::new(true)),
            dropped_after_backend: Arc::new(AtomicBool::new(false)),
            synchronize_calls: AtomicU64::new(0),
            synchronize_failures: AtomicU64::new(0),
            stream_drops: Arc::new(AtomicU64::new(0)),
            allocation_rendezvous: Mutex::new(None),
            close_returned_probe: Mutex::new(None),
            observed_close_return_during_allocation: AtomicBool::new(false),
        }
    }

    fn allocate_calls(&self) -> u64 {
        self.allocate_calls.load(Ordering::Acquire)
    }

    fn fail_on_call(&self, call: u64) {
        self.fail_on_call.store(call, Ordering::Release);
    }

    fn panic_on_call(&self, call: u64) {
        self.panic_on_call.store(call, Ordering::Release);
    }

    fn mismatch_on_call(&self, call: u64) {
        self.mismatch_on_call.store(call, Ordering::Release);
    }

    fn dropped_after_backend_probe(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.dropped_after_backend)
    }

    fn fail_synchronize(&self, failures: u64) {
        self.synchronize_failures.store(failures, Ordering::Release);
    }

    fn synchronize_calls(&self) -> u64 {
        self.synchronize_calls.load(Ordering::Acquire)
    }

    fn stream_drops(&self) -> u64 {
        self.stream_drops.load(Ordering::Acquire)
    }

    fn set_allocation_rendezvous(
        &self,
        entered: Arc<std::sync::Barrier>,
        close_attempting: Arc<std::sync::Barrier>,
    ) {
        *self.allocation_rendezvous.lock().unwrap() = Some((entered, close_attempting));
    }

    fn set_close_returned_probe(&self, probe: Arc<AtomicBool>) {
        *self.close_returned_probe.lock().unwrap() = Some(probe);
    }

    fn observed_close_return_during_allocation(&self) -> bool {
        self.observed_close_return_during_allocation
            .load(Ordering::Acquire)
    }
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = ();
    type Fence = ();
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let rendezvous = self.allocation_rendezvous.lock().unwrap().clone();
        if let Some((entered, close_attempting)) = rendezvous {
            entered.wait();
            close_attempting.wait();
            let probe = self.close_returned_probe.lock().unwrap().clone();
            for _ in 0..256 {
                std::thread::yield_now();
            }
            if probe.is_some_and(|probe| probe.load(Ordering::Acquire)) {
                self.observed_close_return_during_allocation
                    .store(true, Ordering::Release);
            }
        }
        let call = self.allocate_calls.fetch_add(1, Ordering::AcqRel) + 1;
        assert_ne!(
            self.panic_on_call.load(Ordering::Acquire),
            call,
            "injected dynamic-pool allocation panic"
        );
        if self.fail_on_call.load(Ordering::Acquire) == call {
            return Err(TestRuntimeError);
        }
        let request = permit.into_request();
        let mut descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        if self.mismatch_on_call.load(Ordering::Acquire) == call {
            descriptor.size_bytes += 1;
        }
        Ok(TestBuffer {
            descriptor,
            backend_alive: Arc::downgrade(&self.backend_alive),
            dropped_after_backend: Arc::clone(&self.dropped_after_backend),
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(TestStream {
            drops: Arc::clone(&self.stream_drops),
        })
    }

    fn stream_state(&self, _stream: &Self::Stream) -> StreamState {
        StreamState::Ready
    }

    fn encode_copy(
        &self,
        _source: &Self::Buffer,
        _destination: &Self::Buffer,
        _region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn encode_upload(
        &self,
        _source: &[u8],
        _source_layout: HostTransferLayout,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn encode_zero(
        &self,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn submit(
        &self,
        _stream: &mut Self::Stream,
        _commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        Ok(())
    }

    fn query_fence(&self, _fence: &Self::Fence) -> FenceQuery<Self::Error> {
        FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn wait_fence(
        &self,
        _fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), Self::Error> {
        self.synchronize_calls.fetch_add(1, Ordering::AcqRel);
        if self
            .synchronize_failures
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            Err(TestRuntimeError)
        } else {
            Ok(())
        }
    }

    fn readback(
        &self,
        _stream: &mut Self::Stream,
        _source: &Self::Buffer,
        _region: CopyRegion,
        _output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(Vec::new())
    }

    fn describe_error(&self, _error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("dynamic_pool_test", "injected failure", true)
    }
}

#[derive(Clone, Copy)]
enum TestDemand {
    Fixed,
    Tokens,
}

#[derive(Clone)]
struct PoolCatalog {
    pools: Vec<DynamicBackingPoolSpec>,
    descriptors: Vec<DynamicResourceDescriptor>,
    pool_id: DynamicBackingPoolId,
    profile: DynamicStorageProfile,
}

fn canonical_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonical_json).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        scalar => scalar,
    }
}

fn pool_catalog(
    profile: DynamicStorageProfile,
    lifetime: AllocationLifetime,
    layout_digit: char,
    resource_count: usize,
    maximum_resident_bytes: u64,
    demand: TestDemand,
) -> PoolCatalog {
    pool_catalog_with_options(
        profile,
        lifetime,
        layout_digit,
        resource_count,
        maximum_resident_bytes,
        demand,
        "state",
        false,
        StateInitialization::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn pool_catalog_with_options(
    profile: DynamicStorageProfile,
    lifetime: AllocationLifetime,
    layout_digit: char,
    resource_count: usize,
    maximum_resident_bytes: u64,
    demand: TestDemand,
    usage: &str,
    share_step_slot: bool,
    initialization: StateInitialization,
) -> PoolCatalog {
    assert!(!share_step_slot || lifetime == AllocationLifetime::Step && resource_count > 1);
    let layout_fingerprint = layout_digit.to_string().repeat(64);
    let compatibility = json!({
        "version": {"major": 1, "minor": 0},
        "profile": profile,
        "usage": usage,
        "element_type": "u8",
        "logical_layout_fingerprint": layout_fingerprint,
        "alignment_bytes": 16
    });
    let compatibility_bytes = serde_json::to_vec(&canonical_json(compatibility.clone())).unwrap();
    let pool_id_text = format!(
        "dynamic-pool/sha256/{:x}",
        Sha256::digest(compatibility_bytes)
    );
    let pool_id: DynamicBackingPoolId = serde_json::from_value(json!(pool_id_text)).unwrap();
    let mut descriptors = Vec::new();
    let mut resource_ids = Vec::new();
    for index in 0..resource_count {
        let resource_id = format!("resource/dynamic-{layout_digit}-{index:02}");
        resource_ids.push(ResourceId::new(resource_id.clone()).unwrap());
        let demand = match demand {
            TestDemand::Fixed => json!({"fixed": {"bytes": 64}}),
            TestDemand::Tokens => {
                json!({"tokens": {"bytes_per_token": 64, "maximum_tokens": 4}})
            }
        };
        descriptors.push(
            serde_json::from_value(json!({
                "base_resource_id": resource_id,
                "demand": demand,
                "alignment_bytes": 16,
                "usage": usage,
                "element_type": "u8",
                "lifetime": match lifetime {
                    AllocationLifetime::Request => "request",
                    AllocationLifetime::Sequence => "sequence",
                    AllocationLifetime::Step => "step",
                    _ => panic!("test pool supports request/sequence/step lifetimes"),
                },
                "kind": "value",
                "storage": {
                    "profile": profile,
                    "logical_layout_fingerprint": layout_fingerprint
                },
                "pool_id": pool_id_text,
                "initialization": initialization,
                "theoretical_maximum_instances": 64
            }))
            .unwrap(),
        );
    }
    resource_ids.sort();
    descriptors.sort_by(|left: &DynamicResourceDescriptor, right| {
        left.base_resource_id().cmp(right.base_resource_id())
    });
    let minimum = if share_step_slot {
        64
    } else {
        64 * u64::try_from(resource_count).unwrap()
    };
    let theoretical_per_descriptor = match demand {
        TestDemand::Fixed => 64_u128,
        TestDemand::Tokens => 256_u128,
    };
    let theoretical_ceiling =
        theoretical_per_descriptor * 64 * u128::try_from(resource_count).unwrap();
    let step_resource_slots = if lifetime == AllocationLifetime::Step {
        if share_step_slot {
            vec![serde_json::json!({
                "kind": "ordered_single_fence_step_wave",
                "resource_ids": resource_ids
            })]
        } else {
            resource_ids
                .iter()
                .map(|resource_id| {
                    serde_json::json!({
                        "kind": "dedicated",
                        "resource_ids": [resource_id]
                    })
                })
                .collect::<Vec<_>>()
        }
    } else {
        Vec::new()
    };
    let pool: DynamicBackingPoolSpec = serde_json::from_value(json!({
            "pool_id": pool_id_text,
            "compatibility": compatibility,
            "resource_ids": resource_ids,
            "minimum_request_bytes": if lifetime == AllocationLifetime::Request { minimum } else { 0 },
            "minimum_sequence_bytes": if lifetime == AllocationLifetime::Sequence { minimum } else { 0 },
            "minimum_step_bytes": if lifetime == AllocationLifetime::Step { minimum } else { 0 },
            "minimum_invocation_peak_bytes": 0,
            "step_resource_slots": step_resource_slots,
            "theoretical_ceiling_bytes": theoretical_ceiling.to_string(),
            "provisioning": {
                "mode": "demand_driven_elastic",
                "minimum_resident_bytes": minimum,
                "maximum_resident_bytes": maximum_resident_bytes
            },
            "invocation_liveness_mode": "no_invocation_resources",
            "invocation_liveness": []
        }))
        .unwrap();
    PoolCatalog {
        pools: vec![pool],
        descriptors,
        pool_id,
        profile,
    }
}

fn shared_step_activation_catalog(profile: DynamicStorageProfile) -> PoolCatalog {
    pool_catalog_with_options(
        profile,
        AllocationLifetime::Step,
        'c',
        2,
        256,
        TestDemand::Tokens,
        "activations",
        true,
        StateInitialization::None,
    )
}

fn combine_catalogs(catalogs: &[PoolCatalog]) -> PoolCatalog {
    let mut pools = catalogs
        .iter()
        .flat_map(|catalog| catalog.pools.clone())
        .collect::<Vec<_>>();
    pools.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
    let mut descriptors = catalogs
        .iter()
        .flat_map(|catalog| catalog.descriptors.clone())
        .collect::<Vec<_>>();
    descriptors.sort_by(|left, right| left.base_resource_id().cmp(right.base_resource_id()));
    PoolCatalog {
        pools,
        descriptors,
        pool_id: catalogs[0].pool_id.clone(),
        profile: catalogs[0].profile,
    }
}

struct Harness {
    root: Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: Arc<TestRuntime>,
    pool_ids: Vec<DynamicBackingPoolId>,
}

fn new_runtime(catalog: &PoolCatalog, total_memory_bytes: u64) -> Arc<TestRuntime> {
    let ordinal = NEXT_TEST_DEVICE.fetch_add(1, Ordering::Relaxed);
    Arc::new(TestRuntime::new(
        DeviceId::new(format!("device.dynamic-pool-test-{ordinal}")).unwrap(),
        total_memory_bytes,
        catalog
            .pools
            .iter()
            .map(|pool| pool.compatibility().profile())
            .collect(),
    ))
}

fn harness(
    runtime: Arc<TestRuntime>,
    catalog: PoolCatalog,
    usable_capacity_bytes: u64,
    mismatched_coordinator: bool,
) -> Harness {
    harness_with_nodes(
        runtime,
        catalog,
        usable_capacity_bytes,
        mismatched_coordinator,
        Arc::from(Vec::<PlanNode>::new()),
    )
}

fn harness_with_nodes(
    runtime: Arc<TestRuntime>,
    catalog: PoolCatalog,
    usable_capacity_bytes: u64,
    mismatched_coordinator: bool,
    nodes: Arc<[PlanNode]>,
) -> Harness {
    let generation = issue_generation().unwrap();
    let plan_id = PlanId::new(format!("plan/dynamic-pool-test/{generation}")).unwrap();
    let plan_hash: PlanHash = serde_json::from_value(json!("1".repeat(64))).unwrap();
    let request_id =
        RequestIdentity::new(format!("request/dynamic-pool-test/{generation}")).unwrap();
    let pool_identity = ResourcePoolIdentity {
        pool_id: ResourcePoolId::issue(generation).unwrap(),
        plan_id: plan_id.clone(),
        plan_hash: plan_hash.clone(),
        device_id: runtime.descriptor.id.clone(),
        device_runtime_implementation_fingerprint: runtime
            .descriptor
            .runtime_implementation_fingerprint
            .clone(),
        admission_generation: generation,
    };
    let binding = StaticProvisioningBinding {
        pool_identity,
        plan_id,
        plan_hash,
        request_id,
        device_id: runtime.descriptor.id.clone(),
        device_runtime_implementation_fingerprint: runtime
            .descriptor
            .runtime_implementation_fingerprint
            .clone(),
        device_capacity_bytes: runtime.descriptor.total_memory_bytes,
        usable_capacity_bytes,
        plan_static_bytes: 0,
        admitted_bytes: 0,
        maximum_active_sequences: 8,
        admission_generation: generation,
    };
    let account = device_capacity_account(
        binding.device_id(),
        binding.device_runtime_implementation_fingerprint(),
        binding.device_capacity_bytes(),
    )
    .unwrap();
    let (planned_coordinator, domains) = plan_dynamic_pool_admission(
        binding.maximum_active_sequences(),
        &catalog.pools,
        &catalog.descriptors,
    )
    .unwrap();
    let logical_admission = if mismatched_coordinator {
        LogicalAdmissionCoordinator::new(
            vec![(
                CapacityDomainId::new(99).unwrap(),
                CapacityDomainSpec::new(
                    CapacityUnits::ZERO,
                    CapacityUnits::new(usable_capacity_bytes),
                )
                .unwrap(),
            )],
            binding.maximum_active_sequences(),
        )
        .unwrap()
    } else {
        planned_coordinator
    };
    let budget = account.register_budget(usable_capacity_bytes).unwrap();
    let dynamic_pools = Arc::new(
        DynamicPoolSet::new(
            Arc::clone(&runtime),
            binding.clone(),
            budget,
            logical_admission,
            domains,
            nodes,
        )
        .unwrap(),
    );
    let pool_ids = catalog
        .pools
        .iter()
        .map(|pool| pool.pool_id().clone())
        .collect();
    let provisioned = ProvisionedPlanResources::new(StaticProvisioning::NoStatic(NoStatic {
        maintenance_controller: DynamicPoolMaintenanceController::new(Arc::clone(&dynamic_pools)),
        dynamic_pools: Arc::clone(&dynamic_pools),
        binding,
        runtime: Arc::clone(&runtime),
    }));
    let provisioning = provisioned.into_provisioning();
    let StaticProvisioning::NoStatic(no_static) = provisioning else {
        unreachable!()
    };
    let root = no_static.into_plan_runtime();
    Harness {
        root,
        runtime,
        pool_ids,
    }
}

fn linear_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

fn paged_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena { block_bytes: 64 },
        DynamicStorageView::PagedRegions { block_bytes: 64 },
    )
    .unwrap()
}

fn shape(tokens: u64) -> DynamicResourceShape {
    DynamicResourceShape::new(1, tokens, 1).unwrap()
}

fn work(tokens: usize) -> ResourceWorkShape {
    ResourceWorkShape::single(token_span(tokens)).unwrap()
}

fn work_with_ceiling(tokens: usize, maximum_tokens: usize) -> ResourceWorkShape {
    let token_ids = (0..tokens)
        .map(|token| u32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    ResourceWorkShape::single(
        TokenSpanWork::from_token_ids_with_fit(&token_ids, 0..token_ids.len(), maximum_tokens)
            .unwrap(),
    )
    .unwrap()
}

fn token_span(tokens: usize) -> TokenSpanWork {
    let token_ids = (0..tokens)
        .map(|token| u32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    TokenSpanWork::from_token_ids(&token_ids, 0..token_ids.len()).unwrap()
}

fn chunked_work(tokens: usize, immediate_range: std::ops::Range<usize>) -> ResourceWorkShape {
    let token_ids = (0..tokens)
        .map(|token| u32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    ResourceWorkShape::single(TokenSpanWork::from_token_ids(&token_ids, immediate_range).unwrap())
        .unwrap()
}

fn request_admission() -> RequestResourceAdmissionRequest {
    RequestResourceAdmissionRequest::new(
        work(1),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

fn admitted_request(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    suffix: &str,
) -> Arc<AdmittedRequestResources<TestRuntime>> {
    let binding = root.trusted_runtime_binding().unwrap();
    match binding
        .try_admit_request(
            request_admission(),
            RunId::new(format!("run/{suffix}")).unwrap(),
            RequestIdentity::new(format!("request/{suffix}")).unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::Admitted(request) => request,
        _ => panic!("test request must be admitted from resident backing"),
    }
}

fn admitted_sequence(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    suffix: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    admitted_sequence_with_ceiling(root, suffix, 1)
}

fn admitted_sequence_with_ceiling(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    suffix: &str,
    maximum_tokens: usize,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let work = work_with_ceiling(1, maximum_tokens);
    let binding = root.trusted_runtime_binding().unwrap();
    let request = match binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work.clone(),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new(format!("run/{suffix}")).unwrap(),
            RequestIdentity::new(format!("request/{suffix}")).unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::Admitted(request) => request,
        _ => panic!("test request must be admitted from resident backing"),
    };
    let admission = SequenceResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    match request.try_admit_sequence(admission).unwrap() {
        SequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
        _ => panic!("test sequence must be admitted from resident backing"),
    }
}

fn expect_authority_source_rejection<T>(result: Result<T, VNextError>, selected: &str) {
    match result {
        Err(error) => assert!(
            error
                .to_string()
                .contains(&format!("permanently selected for {selected}")),
            "unexpected authority-source error: {error}"
        ),
        Ok(_) => panic!("cross-source sequence authority was unexpectedly admitted"),
    }
}

fn close_dynamic_test_root(root: Arc<PlanRuntimeResources<TestRuntime>>) {
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("dynamic test root retained {strong_count} references")
        }
        Err(failure) => panic!("dynamic test root close failed: {:?}", failure.failure()),
    }
}

fn claim_size(
    pools: &DynamicPoolSet<TestRuntime>,
    pool: &Arc<DynamicBackingPool<TestRuntime>>,
    size_bytes: u64,
) -> LogicalBackingSliceAuthority {
    let request = evaluated_request(pool, size_bytes);
    let BackingPrepareDecision::Prepared(prepared) =
        pools.prepare_claim(std::slice::from_ref(&request)).unwrap()
    else {
        panic!("resident test pool must prepare its exact physical claim")
    };
    prepared.commit().pop().unwrap()
}

fn evaluated_request<'a>(
    pool: &'a Arc<DynamicBackingPool<TestRuntime>>,
    size_bytes: u64,
) -> EvaluatedBackingRequest<'a> {
    evaluated_descriptor_request(pool, 0, size_bytes)
}

fn evaluated_descriptor_request<'a>(
    pool: &'a Arc<DynamicBackingPool<TestRuntime>>,
    descriptor_index: usize,
    size_bytes: u64,
) -> EvaluatedBackingRequest<'a> {
    let descriptor = &pool.domain.descriptors[descriptor_index];
    EvaluatedBackingRequest {
        domain: &pool.domain,
        claim_identity: PhysicalBackingClaimIdentity::new(
            pool.domain.pool_id().clone(),
            vec![descriptor.base_resource_id().clone()],
        )
        .unwrap(),
        size_bytes,
        projections: vec![EvaluatedBackingProjection {
            descriptor,
            physical_offset_bytes: 0,
            size_bytes,
        }],
    }
}

#[test]
fn reused_extent_receives_a_fresh_pending_initialization_authority() {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        64,
        TestDemand::Fixed,
        "state",
        false,
        StateInitialization::Zero,
    );
    let runtime = new_runtime(&catalog, 64);
    let harness = harness(runtime, catalog, 64, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);

    let first = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let first_segment = first.evidence().segments()[0].clone();
    let first_generation = first.evidence().segment_generation();
    let first_cell = Arc::clone(first.initialization_cell().unwrap());
    assert_eq!(
        first.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Pending)
    );
    assert!(first_cell.prepare("wave/a").unwrap());
    first_cell.mark_in_flight("wave/a").unwrap();
    first_cell.finish("wave/a", true).unwrap();
    assert_eq!(
        first.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Initialized)
    );
    let retained = first.retained();
    assert!(Arc::ptr_eq(
        retained.initialization_cell().unwrap(),
        &first_cell
    ));
    drop(first);
    assert_eq!(
        retained.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Initialized)
    );
    drop(retained);

    let second = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let second_segment = &second.evidence().segments()[0];
    assert_eq!(second_segment.chunk(), first_segment.chunk());
    assert_eq!(second_segment.offset_bytes(), first_segment.offset_bytes());
    assert_ne!(second.evidence().segment_generation(), first_generation);
    assert!(!Arc::ptr_eq(
        second.initialization_cell().unwrap(),
        &first_cell
    ));
    assert_eq!(
        second.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Pending)
    );
}

#[test]
fn failed_initialization_poison_is_scoped_to_one_extent_generation() {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        64,
        TestDemand::Fixed,
        "state",
        false,
        StateInitialization::Zero,
    );
    let runtime = new_runtime(&catalog, 64);
    let harness = harness(runtime, catalog, 64, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);

    let failed = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let failed_generation = failed.evidence().segment_generation();
    let failed_segment = failed.evidence().segments()[0].clone();
    let failed_cell = Arc::clone(failed.initialization_cell().unwrap());
    assert!(failed_cell.prepare("wave/failed").unwrap());
    failed_cell.mark_in_flight("wave/failed").unwrap();
    failed_cell.finish("wave/failed", false).unwrap();
    assert_eq!(
        failed.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Poisoned)
    );
    drop(failed);

    let recovered = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let recovered_segment = &recovered.evidence().segments()[0];
    assert_eq!(recovered_segment.chunk(), failed_segment.chunk());
    assert_eq!(
        recovered_segment.offset_bytes(),
        failed_segment.offset_bytes()
    );
    assert_ne!(recovered.evidence().segment_generation(), failed_generation);
    assert!(!Arc::ptr_eq(
        recovered.initialization_cell().unwrap(),
        &failed_cell
    ));
    assert_eq!(
        recovered.initialization_status().unwrap(),
        Some(BackingInitializationStatus::Pending)
    );
}

#[test]
fn logical_projection_range_crosses_physical_chunks_without_prefix_aliasing() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Request,
        'd',
        1,
        256,
        TestDemand::Tokens,
    );
    let segments = vec![
        BackingSegment::from_chunk(&catalog.pool_id, 1, 7, 0, 64).unwrap(),
        BackingSegment::from_chunk(&catalog.pool_id, 2, 8, 0, 64).unwrap(),
    ];

    let projection = backing_segment_range(&segments, 32, 80).unwrap();

    assert_eq!(projection.len(), 2);
    assert_eq!(projection[0].chunk_ordinal(), 1);
    assert_eq!(projection[0].offset_bytes(), 32);
    assert_eq!(projection[0].length_bytes(), 32);
    assert_eq!(projection[1].chunk_ordinal(), 2);
    assert_eq!(projection[1].offset_bytes(), 0);
    assert_eq!(projection[1].length_bytes(), 48);
    assert!(backing_segment_range(&segments, 96, 64).is_err());
}

struct CleanupPressureTask {
    ready: Arc<AtomicBool>,
}

impl DeferredDeviceCleanupTask for CleanupPressureTask {
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        if self.ready.load(Ordering::Acquire) {
            DeferredDeviceCleanupDisposition::Completed
        } else {
            DeferredDeviceCleanupDisposition::Retryable
        }
    }
}

#[test]
fn saturated_cleanup_backlog_blocks_new_authority_until_maintenance() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(runtime, catalog, 512, false);
    assert!(harness.root.maintain_deferred_cleanups(0).is_err());
    assert!(
        harness
            .root
            .maintain_deferred_cleanups(
                crate::vnext::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS + 1,
            )
            .is_err()
    );
    let ready = Arc::new(AtomicBool::new(false));
    for _ in 0..crate::vnext::MAX_DEFERRED_DEVICE_CLEANUP_TASKS {
        defer_device_cleanup(
            harness.root.deferred_cleanup_domain,
            CleanupPressureTask {
                ready: Arc::clone(&ready),
            },
        );
    }

    assert!(harness.root.deferred_cleanup_status().is_saturated());
    assert!(harness.root.trusted_runtime_binding().is_err());
    ready.store(true, Ordering::Release);
    let receipt = harness
        .root
        .maintain_deferred_cleanups(crate::vnext::MAX_DEFERRED_DEVICE_CLEANUP_TASKS)
        .unwrap();
    assert_eq!(
        receipt.completed(),
        crate::vnext::MAX_DEFERRED_DEVICE_CLEANUP_TASKS
    );
    assert_eq!(receipt.status_after().pending(), 0);
    drop(harness.root.trusted_runtime_binding().unwrap());
    close_dynamic_test_root(harness.root);
}

#[test]
fn zero_initial_capacity_defers_until_typed_initialization() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(runtime, catalog, 512, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let decision = binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/zero-initial").unwrap(),
            RequestIdentity::new("request/zero-initial").unwrap(),
        )
        .unwrap();
    let RequestResourceAdmissionDecision::BackingDeferred(deferred) = decision else {
        panic!("zero-resident pool must defer")
    };
    assert_eq!(deferred.evidence().blockers().len(), 1);
    assert_eq!(
        deferred.evidence().blockers()[0].reason(),
        DynamicBackingDeferralReason::GrowthRequired
    );
    assert_eq!(harness.runtime.allocate_calls(), 0);

    let maintenance = &harness.root.maintenance_controller;
    let receipt = maintenance
        .initialize_pool(&harness.pool_ids[0])
        .unwrap()
        .expect("first initialization grows");
    assert_eq!(receipt.published_capacity_bytes(), 64);
    assert!(maintenance
        .initialize_pool(&harness.pool_ids[0])
        .unwrap()
        .is_none());
    assert_eq!(harness.runtime.allocate_calls(), 1);
}

#[test]
fn backing_deferral_reports_the_whole_pool_shortfall() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        3,
        192,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(runtime, catalog, 192, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let admission = || {
        binding
            .try_admit_request(
                request_admission(),
                RunId::new("run/batch-shortfall").unwrap(),
                RequestIdentity::new("request/batch-shortfall").unwrap(),
            )
            .unwrap()
    };
    let RequestResourceAdmissionDecision::BackingDeferred(deferred) = admission() else {
        panic!("zero-resident pool must expose its complete batch shortfall")
    };
    assert_eq!(deferred.evidence().blockers().len(), 1);
    assert_eq!(deferred.evidence().blockers()[0].requested_bytes(), 192);
    assert_eq!(deferred.evidence().blockers()[0].free_bytes(), 0);

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = deferred.maintain().unwrap()
    else {
        panic!("current batch shortfall must grow its pool")
    };
    assert_eq!(receipt.growths().len(), 1);
    assert_eq!(receipt.growths()[0].chunk_bytes(), 192);
    assert_eq!(harness.runtime.allocate_calls(), 1);
    assert!(matches!(
        admission(),
        RequestResourceAdmissionDecision::Admitted(_)
    ));
}

#[test]
fn logical_fit_deferral_grows_unclaimed_backing_capacity() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let request = match binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work_with_ceiling(1, 4),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/logical-fit-growth").unwrap(),
            RequestIdentity::new("request/logical-fit-growth").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::Admitted(request) => request,
        _ => panic!("test request must be admitted from resident backing"),
    };
    let admission = SequenceResourceAdmissionRequest::new(
        chunked_work(4, 0..1),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let SequenceResourceAdmissionDecision::Deferred(deferred) =
        request.try_admit_sequence(admission.clone()).unwrap()
    else {
        panic!("full-fit capacity above current residency must request logical growth")
    };
    assert_eq!(deferred.action(), DeferredAction::AwaitBackingGrowth);
    assert_eq!(deferred.blockers().len(), 1);
    assert_eq!(deferred.blockers()[0].requested().get(), 256);
    assert_eq!(deferred.blockers()[0].current_total().get(), 64);

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = harness
        .root
        .maintain_for_admission_deferred(&deferred)
        .unwrap()
    else {
        panic!("current logical growth deferral must materialize fit capacity")
    };
    assert_eq!(receipt.growths().len(), 1);
    assert_eq!(receipt.growths()[0].chunk_bytes(), 192);
    assert_eq!(receipt.growths()[0].published_capacity_bytes(), 256);
    assert!(matches!(
        request.try_admit_sequence(admission).unwrap(),
        SequenceResourceAdmissionDecision::Admitted(_)
    ));
}

#[test]
fn full_plan_budget_returns_typed_wait_and_reuses_backing_after_availability_change() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 64);
    let harness = harness(Arc::clone(&runtime), catalog, 64, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let held_backing = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let deferred = match binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/capacity-wait-second").unwrap(),
            RequestIdentity::new("request/capacity-wait-second").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("occupied resident backing must ask for physical growth"),
    };
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .epochs()
        .unwrap();
    let churned = harness
        .root
        .dynamic_pools
        .logical_admission
        .notify_domain_availability_changed(pool.domain.domain_id)
        .unwrap();
    assert_ne!(churned, before);
    let allocations_before = runtime.allocate_calls();

    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        current_epochs,
        pressure,
        ..
    } = deferred.maintain().unwrap()
    else {
        panic!("unrelated epoch churn must preserve typed request backing pressure")
    };
    assert_eq!(current_epochs, churned);
    assert_eq!(pressure.scope(), &DeviceCapacityPressureScope::PlanBudget);
    assert_eq!(pressure.requested_bytes(), 64);
    assert_eq!(pressure.available_bytes(), 0);
    assert_eq!(runtime.allocate_calls(), allocations_before);
    let status = harness.root.maintenance_controller.status().unwrap();
    assert_eq!(status.process_claimed_bytes(), 64);
    assert_eq!(status.budget_claimed_bytes(), 64);
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);

    drop(held_backing);
    let after_release = harness
        .root
        .dynamic_pools
        .logical_admission
        .epochs()
        .unwrap();
    assert!(after_release.capacity_epoch() > before.capacity_epoch());
    assert!(matches!(
        binding
            .try_admit_request(
                request_admission(),
                RunId::new("run/capacity-wait-second-retry").unwrap(),
                RequestIdentity::new("request/capacity-wait-second-retry").unwrap(),
            )
            .unwrap(),
        RequestResourceAdmissionDecision::Admitted(_)
    ));
    assert_eq!(runtime.allocate_calls(), allocations_before);
}

#[test]
fn theoretical_pool_ceiling_remains_terminal_after_device_budget_accepts_growth() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'b',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&harness.pool_ids[0]).unwrap();
    let allocations_before = runtime.allocate_calls();

    let error = maintenance
        .grow_pool(&harness.pool_ids[0], 64)
        .expect_err("theoretical pool ceiling must remain fail-closed");
    assert!(error
        .to_string()
        .contains("dynamic pool growth exceeds its core-derived resident maximum"));
    let status = maintenance.status().unwrap();
    assert_eq!(status.budget_claimed_bytes(), 64);
    assert_eq!(status.process_claimed_bytes(), 64);
    assert_eq!(status.pools()[0].resident_bytes(), 64);
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);
    assert_eq!(runtime.allocate_calls(), allocations_before);
}

#[test]
fn plan_budget_pressure_rebalances_idle_chunks_across_pools() {
    let donor_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '2',
        1,
        128,
        TestDemand::Fixed,
    );
    let target_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '3',
        1,
        128,
        TestDemand::Fixed,
    );
    let donor_pool_id = donor_catalog.pool_id.clone();
    let target_pool_id = target_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[donor_catalog, target_catalog]);
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(Arc::clone(&runtime), catalog, 192, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&donor_pool_id).unwrap();
    let donor_growth = maintenance.grow_pool(&donor_pool_id, 64).unwrap();
    maintenance.initialize_pool(&target_pool_id).unwrap();
    let target_pool = Arc::clone(&harness.root.dynamic_pools.pools[&target_pool_id]);
    let held_target = claim_size(&harness.root.dynamic_pools, &target_pool, 64);
    let request = evaluated_request(&target_pool, 64);
    let BackingPrepareDecision::Deferred(deferred) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("occupied target backing must defer under the exact plan budget")
    };
    let device_epochs_before = harness
        .root
        .dynamic_pools
        .budget
        .availability_snapshot()
        .unwrap();

    let DynamicDeferredMaintenanceOutcome::Maintained(growth) =
        maintenance.maintain_for_live_deferred(&deferred).unwrap()
    else {
        panic!("idle donor residency must be rebalanced into the blocked target pool")
    };
    assert_eq!(growth.growths().len(), 1);
    assert_eq!(growth.growths()[0].pool_id(), &target_pool_id);
    assert_eq!(growth.growths()[0].chunk_bytes(), 64);
    let rebalance = growth
        .rebalance()
        .expect("cross-pool maintenance must expose its reclaim receipt");
    assert_eq!(rebalance.reclaimed_chunks(), 1);
    assert_eq!(rebalance.reclaimed_bytes(), 64);
    assert_eq!(rebalance.pools().len(), 1);
    assert_eq!(rebalance.pools()[0].pool_id(), &donor_pool_id);
    assert_eq!(
        rebalance.pools()[0].chunks(),
        std::slice::from_ref(donor_growth.chunk())
    );
    assert_eq!(rebalance.pools()[0].published_capacity_bytes(), 64);
    let serialized_rebalance = serde_json::to_value(rebalance).unwrap();
    assert_eq!(
        serialized_rebalance["pools"][0]["pool_id"],
        json!(donor_pool_id.as_str())
    );
    assert_eq!(
        serialized_rebalance["pools"][0]["chunks"][0]["ordinal"],
        json!(donor_growth.chunk().ordinal())
    );
    assert_eq!(
        serialized_rebalance["pools"][0]["chunks"][0]["generation"],
        json!(donor_growth.chunk().generation())
    );
    assert!(rebalance.plan_device_capacity_epoch() > device_epochs_before.plan_epoch());
    assert!(rebalance.process_device_capacity_epoch() > device_epochs_before.process_epoch());

    let status = maintenance.status().unwrap();
    let donor = status
        .pools()
        .iter()
        .find(|pool| pool.pool_id() == &donor_pool_id)
        .unwrap();
    let target = status
        .pools()
        .iter()
        .find(|pool| pool.pool_id() == &target_pool_id)
        .unwrap();
    assert_eq!(donor.resident_bytes(), 64);
    assert_eq!(donor.resident_chunks(), 1);
    assert_eq!(target.resident_bytes(), 128);
    assert_eq!(target.resident_chunks(), 2);
    assert_eq!(status.budget_claimed_bytes(), 192);
    assert_eq!(status.process_claimed_bytes(), 192);
    assert_eq!(runtime.allocate_calls(), 4);

    drop(held_target);
    let BackingPrepareDecision::Prepared(prepared) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("rebalanced target pool must satisfy the original claim")
    };
    drop(prepared.commit());
}

#[test]
fn live_idle_donor_boundary_waits_then_rebalances_after_exact_release() {
    let donor_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '4',
        1,
        128,
        TestDemand::Fixed,
    );
    let target_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '5',
        1,
        128,
        TestDemand::Fixed,
    );
    let donor_pool_id = donor_catalog.pool_id.clone();
    let target_pool_id = target_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[donor_catalog, target_catalog]);
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(runtime, catalog, 192, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&donor_pool_id).unwrap();
    maintenance.grow_pool(&donor_pool_id, 64).unwrap();
    maintenance.initialize_pool(&target_pool_id).unwrap();
    let donor_pool = Arc::clone(&harness.root.dynamic_pools.pools[&donor_pool_id]);
    let target_pool = Arc::clone(&harness.root.dynamic_pools.pools[&target_pool_id]);
    let donor_minimum = claim_size(&harness.root.dynamic_pools, &donor_pool, 64);
    let donor_excess = claim_size(&harness.root.dynamic_pools, &donor_pool, 64);
    let held_target = claim_size(&harness.root.dynamic_pools, &target_pool, 64);
    let request = evaluated_request(&target_pool, 64);
    let BackingPrepareDecision::Deferred(deferred) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("full target pool must defer")
    };

    assert!(matches!(
        maintenance.maintain_for_live_deferred(&deferred).unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
    ));
    let blocked = maintenance.status().unwrap();
    assert_eq!(blocked.budget_claimed_bytes(), 192);
    assert_eq!(
        blocked
            .pools()
            .iter()
            .find(|pool| pool.pool_id() == &donor_pool_id)
            .unwrap()
            .resident_chunks(),
        2
    );

    drop(donor_excess);
    let DynamicDeferredMaintenanceOutcome::Maintained(growth) =
        maintenance.maintain_for_live_deferred(&deferred).unwrap()
    else {
        panic!("the exact donor release must make one whole chunk reclaimable")
    };
    assert_eq!(growth.rebalance().unwrap().reclaimed_chunks(), 1);
    assert_eq!(growth.rebalance().unwrap().reclaimed_bytes(), 64);
    assert_eq!(maintenance.status().unwrap().budget_claimed_bytes(), 192);

    drop(donor_minimum);
    drop(held_target);
}

#[test]
fn insufficient_idle_reclaim_keeps_all_residency_and_returns_typed_wait() {
    let donor_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '6',
        1,
        96,
        TestDemand::Fixed,
    );
    let target_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '7',
        1,
        128,
        TestDemand::Fixed,
    );
    let donor_pool_id = donor_catalog.pool_id.clone();
    let target_pool_id = target_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[donor_catalog, target_catalog]);
    let runtime = new_runtime(&catalog, 160);
    let harness = harness(runtime, catalog, 160, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&donor_pool_id).unwrap();
    maintenance.grow_pool(&donor_pool_id, 32).unwrap();
    maintenance.initialize_pool(&target_pool_id).unwrap();
    let target_pool = Arc::clone(&harness.root.dynamic_pools.pools[&target_pool_id]);
    let held_target = claim_size(&harness.root.dynamic_pools, &target_pool, 64);
    let request = evaluated_request(&target_pool, 64);
    let BackingPrepareDecision::Deferred(deferred) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("full target pool must defer")
    };
    let before = maintenance.status().unwrap();
    let device_epochs_before = harness
        .root
        .dynamic_pools
        .budget
        .availability_snapshot()
        .unwrap();

    let DynamicDeferredMaintenanceOutcome::WaitForRelease { pressure, .. } =
        maintenance.maintain_for_live_deferred(&deferred).unwrap()
    else {
        panic!("a partial donor chunk must not be reclaimed without satisfying the deficit")
    };
    assert_eq!(pressure.requested_bytes(), 64);
    assert_eq!(pressure.available_bytes(), 0);
    let after = maintenance.status().unwrap();
    assert_eq!(after.pools(), before.pools());
    assert_eq!(after.budget_claimed_bytes(), before.budget_claimed_bytes());
    assert_eq!(
        after.process_claimed_bytes(),
        before.process_claimed_bytes()
    );
    let device_epochs_after = harness
        .root
        .dynamic_pools
        .budget
        .availability_snapshot()
        .unwrap();
    assert_eq!(device_epochs_after, device_epochs_before);

    drop(held_target);
}

#[test]
fn initial_bundle_waits_without_partial_request_and_allows_smaller_bypass() {
    let request_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '8',
        1,
        256,
        TestDemand::Tokens,
    );
    let sequence_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        '9',
        1,
        256,
        TestDemand::Tokens,
    );
    let request_pool_id = request_catalog.pool_id.clone();
    let sequence_pool_id = sequence_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[request_catalog, sequence_catalog]);
    let runtime = new_runtime(&catalog, 320);
    let harness = harness(runtime, catalog, 320, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&request_pool_id).unwrap();
    maintenance.grow_pool(&request_pool_id, 128).unwrap();
    maintenance.initialize_pool(&sequence_pool_id).unwrap();
    maintenance.grow_pool(&sequence_pool_id, 64).unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();

    let work_a = work_with_ceiling(1, 1);
    let active = match binding
        .try_admit_initial_sequence(
            RequestResourceAdmissionRequest::new(
                work_a.clone(),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            SequenceResourceAdmissionRequest::new(
                work_a,
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/initial-bundle-a").unwrap(),
            RequestIdentity::new("request/initial-bundle-a").unwrap(),
        )
        .unwrap()
    {
        InitialSequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
        _ => panic!("A must occupy one request and sequence slice"),
    };
    let before_b = maintenance.status().unwrap();
    let logical_before_b = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();

    let work_b = work_with_ceiling(2, 2);
    let deferred = match binding
        .try_admit_initial_sequence(
            RequestResourceAdmissionRequest::new(
                work_b.clone(),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            SequenceResourceAdmissionRequest::new(
                work_b,
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/initial-bundle-b").unwrap(),
            RequestIdentity::new("request/initial-bundle-b").unwrap(),
        )
        .unwrap()
    {
        InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("B must wait for its larger sequence backing"),
    };
    assert_eq!(
        deferred.evidence().scope(),
        DynamicBackingClaimScope::InitialSequenceBundle
    );
    let request_domain = harness.root.dynamic_pools.pools[&request_pool_id]
        .domain
        .domain_id;
    let sequence_domain = harness.root.dynamic_pools.pools[&sequence_pool_id]
        .domain
        .domain_id;
    assert_eq!(
        deferred
            .evidence()
            .protected_immediate()
            .units_for(request_domain)
            .unwrap()
            .get(),
        128
    );
    assert_eq!(
        deferred
            .evidence()
            .protected_immediate()
            .units_for(sequence_domain)
            .unwrap()
            .get(),
        128
    );
    let after_b = maintenance.status().unwrap();
    let logical_after_b = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(after_b.pools(), before_b.pools());
    assert_eq!(logical_after_b.domains(), logical_before_b.domains());
    assert_eq!(logical_after_b.active_requests(), 1);
    assert_eq!(logical_after_b.active_sequences(), 1);
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
    ));

    let work_c = work_with_ceiling(1, 1);
    let bypass = match binding
        .try_admit_initial_sequence(
            RequestResourceAdmissionRequest::new(
                work_c.clone(),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            SequenceResourceAdmissionRequest::new(
                work_c,
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/initial-bundle-c").unwrap(),
            RequestIdentity::new("request/initial-bundle-c").unwrap(),
        )
        .unwrap()
    {
        InitialSequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
        _ => panic!("C must bypass B while A remains active"),
    };
    let with_c = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(with_c.active_requests(), 2);
    assert_eq!(with_c.active_sequences(), 2);

    drop(deferred);
    drop(bypass);
    drop(active);
    drop(binding);
    close_dynamic_test_root(harness.root);
}

#[test]
fn plan_budget_wait_retains_staged_parent_and_reuses_released_sequence_backing() {
    let request_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        '2',
        1,
        128,
        TestDemand::Fixed,
    );
    let sequence_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        '3',
        1,
        128,
        TestDemand::Fixed,
    );
    let request_pool_id = request_catalog.pool_id.clone();
    let sequence_pool_id = sequence_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[request_catalog, sequence_catalog]);
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(Arc::clone(&runtime), catalog, 192, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&request_pool_id)
        .unwrap();
    harness
        .root
        .maintenance_controller
        .initialize_pool(&sequence_pool_id)
        .unwrap();
    harness
        .root
        .maintenance_controller
        .grow_pool(&request_pool_id, 64)
        .unwrap();

    let active = admitted_sequence(&harness.root, "capacity-wait-active");
    let staged = admitted_request(&harness.root, "capacity-wait-staged");
    let admission = SequenceResourceAdmissionRequest::new(
        work(1),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let SequenceResourceAdmissionDecision::BackingDeferred(deferred) =
        staged.try_admit_sequence(admission.clone()).unwrap()
    else {
        panic!("active sequence must occupy the staged request's sequence backing")
    };
    assert!(harness
        .root
        .trusted_runtime_binding()
        .unwrap()
        .maintain_request_backing_for_deferred(deferred.evidence())
        .is_err());
    let churned = harness
        .root
        .dynamic_pools
        .logical_admission
        .notify_domain_availability_changed(
            harness.root.dynamic_pools.pools[&request_pool_id]
                .domain
                .domain_id,
        )
        .unwrap();
    assert_ne!(churned, deferred.evidence().epochs());
    let allocations_before = runtime.allocate_calls();
    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        current_epochs,
        pressure,
        ..
    } = deferred.maintain().unwrap()
    else {
        panic!("retained parent must tolerate unrelated epoch churn")
    };
    assert_eq!(current_epochs, churned);
    assert_eq!(pressure.scope(), &DeviceCapacityPressureScope::PlanBudget);
    assert_eq!(runtime.allocate_calls(), allocations_before);

    let retained = deferred.into_parent();
    assert!(Arc::ptr_eq(&retained, &staged));
    drop(active);
    let released_epochs = harness
        .root
        .dynamic_pools
        .logical_admission
        .epochs()
        .unwrap();
    assert!(released_epochs.capacity_epoch() > current_epochs.capacity_epoch());
    let admitted = match retained.try_admit_sequence(admission).unwrap() {
        SequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
        _ => panic!("retained staged parent must reuse released sequence backing"),
    };
    assert_eq!(runtime.allocate_calls(), allocations_before);

    drop(admitted);
    drop(retained);
    drop(staged);
    close_dynamic_test_root(harness.root);
}

#[test]
fn capacity_waiter_retains_root_and_close_rejects_new_work() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(runtime, catalog, 512, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let deferred = match binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/close-waiter").unwrap(),
            RequestIdentity::new("request/close-waiter").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident pool must defer before waiter registration"),
    };
    let waiter = deferred.register_waiter().unwrap();
    drop(binding);

    let first_close = match PlanRuntimeResources::close(harness.root) {
        Ok(outcome) => outcome,
        Err(failure) => panic!("no-static root close failed: {:?}", failure.failure()),
    };
    let root = match first_close {
        PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            ..
        } => {
            assert_eq!(strong_count, 3);
            resources
        }
        PlanRuntimeCloseOutcome::Closed(_) => {
            panic!("capacity waiter must retain the owning root")
        }
    };
    assert!(root.is_closing());
    assert!(root.trusted_runtime_binding().is_err());
    assert!(deferred.maintain().is_err());
    assert!(matches!(
        waiter.recheck(),
        Err(VNextError::InvalidExecutionPlan { reason })
            if reason == "closing plan runtime cannot recheck a capacity waiter"
    ));
    drop(waiter);
    drop(deferred);

    let final_close = match PlanRuntimeResources::close(root) {
        Ok(outcome) => outcome,
        Err(failure) => panic!("no-static root close retry failed: {:?}", failure.failure()),
    };
    match final_close {
        PlanRuntimeCloseOutcome::Closed(receipt) => {
            assert_eq!(receipt.released_static_resources(), 0)
        }
        PlanRuntimeCloseOutcome::Referenced { .. } => {
            panic!("released waiter must allow the owning root to close")
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn closing_root_wakes_capacity_waiter_without_lost_notification() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'b',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(runtime, catalog, 512, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let deferred = match binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/async-close-waiter").unwrap(),
            RequestIdentity::new("request/async-close-waiter").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident pool must defer before async waiter registration"),
    };
    let waiter = deferred.register_waiter().unwrap();
    drop(binding);
    let task = tokio::spawn(waiter.wait_for_change());

    let root = match PlanRuntimeResources::close(harness.root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("live async waiter unexpectedly allowed root close")
        }
        Err(failure) => panic!("no-static root close failed: {:?}", failure.failure()),
    };
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .expect("close must wake the capacity waiter within the bounded timeout")
        .expect("capacity waiter task must not panic");
    assert!(matches!(
        result,
        Err(VNextError::InvalidExecutionPlan { reason })
            if reason == "closing plan runtime cancelled its capacity waiter"
    ));
    drop(deferred);

    let final_close = match PlanRuntimeResources::close(root) {
        Ok(outcome) => outcome,
        Err(failure) => panic!("final waiter close failed: {:?}", failure.failure()),
    };
    match final_close {
        PlanRuntimeCloseOutcome::Closed(receipt) => {
            assert_eq!(receipt.released_static_resources(), 0)
        }
        PlanRuntimeCloseOutcome::Referenced { .. } => {
            panic!("woken waiter retained the root after its future completed")
        }
    }
}

#[test]
fn last_sequence_owner_sync_failure_retains_exact_resources_for_retry() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'f',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "last-owner-sync-failure");
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    drop(permit);
    runtime.fail_synchronize(1);

    drop(sequence);
    drop(stream);
    assert_eq!(runtime.synchronize_calls(), 0);
    assert_eq!(runtime.stream_drops(), 0);
    assert_eq!(harness.root.deferred_cleanup_status().pending(), 1);
    let first_cleanup = harness.root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first_cleanup.retryable(), 1);
    assert_eq!(first_cleanup.status_after().pending(), 1);
    assert_eq!(runtime.synchronize_calls(), 1);
    assert_eq!(runtime.stream_drops(), 0);
    let status = harness.root.maintenance_controller.status().unwrap();
    assert_eq!(status.pools()[0].live_segments(), 1);
    assert_eq!(status.budget_claimed_bytes(), 64);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_sequences(),
        1
    );

    let resources = match PlanRuntimeResources::close(harness.root) {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            deferred_cleanup,
        }) => {
            assert!(strong_count >= 2);
            assert!(resources.is_closing());
            assert_eq!(deferred_cleanup.pending(), 1);
            resources
        }
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("failed last-owner recovery released its leaked backing/root")
        }
        Err(failure) => panic!("referenced close failed: {:?}", failure.failure()),
    };
    let second_cleanup = resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(second_cleanup.completed(), 1);
    assert_eq!(second_cleanup.status_after().pending(), 0);
    assert_eq!(runtime.synchronize_calls(), 2);
    assert_eq!(runtime.stream_drops(), 1);
    assert_eq!(
        resources
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_sequences(),
        0
    );
    close_dynamic_test_root(resources);
}

#[test]
fn lifecycle_gate_linearizes_maintenance_before_close() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'c',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(Arc::clone(&runtime), catalog, 512, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let deferred = match binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/lifecycle-race").unwrap(),
            RequestIdentity::new("request/lifecycle-race").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident pool must defer before lifecycle race"),
    };
    drop(binding);

    let entered = Arc::new(std::sync::Barrier::new(DYNAMIC_POOL_CONCURRENT_WORKERS + 1));
    let close_attempting = Arc::new(std::sync::Barrier::new(DYNAMIC_POOL_CONCURRENT_WORKERS + 1));
    let close_returned = Arc::new(AtomicBool::new(false));
    runtime.set_allocation_rendezvous(Arc::clone(&entered), Arc::clone(&close_attempting));
    runtime.set_close_returned_probe(Arc::clone(&close_returned));

    let root = harness.root;
    let (maintenance, close) = std::thread::scope(|scope| {
        let close_root = Arc::clone(&root);
        let close_returned = Arc::clone(&close_returned);
        let worker = std::thread::Builder::new()
            .name("vnext-plan-runtime-close-racer".to_owned())
            .spawn_scoped(scope, move || {
                entered.wait();
                close_attempting.wait();
                let close = PlanRuntimeResources::close(close_root);
                close_returned.store(true, Ordering::Release);
                close
            })
            .expect("the single bounded close-race worker starts");
        let maintenance = deferred.maintain();
        let close = worker
            .join()
            .expect("the bounded close-race worker does not panic");
        (maintenance, close)
    });
    assert!(matches!(
        maintenance.unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    assert!(!runtime.observed_close_return_during_allocation());
    let closing_root = match close {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("caller root must retain the plan after the racing close")
        }
        Err(failure) => panic!("racing close failed: {:?}", failure.failure()),
    };
    assert!(closing_root.is_closing());
    assert!(closing_root.trusted_runtime_binding().is_err());
    assert!(deferred.maintain().is_err());
    drop(deferred);
    drop(root);
    match PlanRuntimeResources::close(closing_root) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(PlanRuntimeCloseOutcome::Referenced { .. }) => {
            panic!("completed maintenance retained an unexpected root reference")
        }
        Err(failure) => panic!("final close failed: {:?}", failure.failure()),
    }
}

#[test]
fn quiescent_sequence_abort_is_one_terminal_transition() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'f',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "quiescent-atomic-abort");
    let session = sequence.open_session().unwrap();

    let terminal = session.try_abort_if_quiescent().unwrap();

    assert_eq!(
        terminal.disposition(),
        SequenceSessionTerminalDisposition::Aborted
    );
    assert!(session.request_cancel().is_err());
    drop(terminal);
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_session_source_permanently_rejects_legacy_activation() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'd',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "session-source-seal");
    let mut stream = sequence.create_execution_stream().unwrap();
    let session = sequence.open_session().unwrap();

    expect_authority_source_rejection(sequence.activate(&mut stream), "sequence sessions");
    session.request_cancel().unwrap();
    let terminal = session.try_abort().unwrap();
    assert_eq!(
        terminal.disposition(),
        SequenceSessionTerminalDisposition::Aborted
    );
    expect_authority_source_rejection(sequence.activate(&mut stream), "sequence sessions");

    drop(terminal);
    drop(session);
    drop(stream);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn legacy_source_permanently_rejects_sessions_after_every_terminal_path() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'e',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();

    let completed_sequence = admitted_sequence(&harness.root, "legacy-completed-source-seal");
    let mut completed_stream = completed_sequence.create_execution_stream().unwrap();
    let permit = completed_sequence.activate(&mut completed_stream).unwrap();
    expect_authority_source_rejection(completed_sequence.open_session(), "legacy streams");
    let completion = permit.synchronize().unwrap().complete().unwrap();
    expect_authority_source_rejection(completed_sequence.open_session(), "legacy streams");
    drop(completion);
    drop(completed_stream);
    drop(completed_sequence);

    let abandoned_sequence = admitted_sequence(&harness.root, "legacy-abandoned-source-seal");
    let mut abandoned_stream = abandoned_sequence.create_execution_stream().unwrap();
    let permit = abandoned_sequence.activate(&mut abandoned_stream).unwrap();
    drop(permit);
    expect_authority_source_rejection(abandoned_sequence.open_session(), "legacy streams");
    drop(abandoned_stream);
    let recovery = abandoned_sequence.recover_abandoned_sequence().unwrap();
    assert_eq!(
        recovery.disposition(),
        ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
    );
    drop(recovery);
    drop(abandoned_sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_authority_source_race_admits_exactly_one_source() {
    assert_eq!(DYNAMIC_POOL_CONCURRENT_WORKERS, 1);
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'f',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "authority-source-race");
    let mut stream = sequence.create_execution_stream().unwrap();
    let start = Arc::new((Mutex::new(false), std::sync::Condvar::new()));

    std::thread::scope(|scope| {
        let worker_sequence = Arc::clone(&sequence);
        let worker_start = Arc::clone(&start);
        let worker = std::thread::Builder::new()
            .name("vnext-sequence-authority-source-racer".to_owned())
            .spawn_scoped(scope, move || {
                let (lock, wake) = &*worker_start;
                let mut started = lock
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while !*started {
                    started = wake
                        .wait(started)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                drop(started);
                worker_sequence.open_session()
            })
            .expect("the single bounded authority-source worker starts");
        let (lock, wake) = &*start;
        let mut started = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *started = true;
        wake.notify_one();
        drop(started);

        let activation = sequence.activate(&mut stream);
        let session = worker
            .join()
            .expect("the bounded authority-source worker does not panic");
        match (activation, session) {
            (Ok(permit), Err(error)) => {
                assert!(error.to_string().contains("legacy streams"));
                let completion = permit.synchronize().unwrap().complete().unwrap();
                drop(completion);
            }
            (Err(error), Ok(session)) => {
                assert!(error.to_string().contains("sequence sessions"));
                session.request_cancel().unwrap();
                let terminal = session.try_abort().unwrap();
                drop(terminal);
                drop(session);
            }
            (Ok(_), Ok(_)) => panic!("both sequence authority sources won one race"),
            (Err(legacy), Err(session)) => panic!(
                "both sequence authority sources lost one race: legacy={legacy}; session={session}"
            ),
        }
    });

    drop(stream);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn poisoned_authority_source_selector_is_fail_closed() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "authority-source-poison");
    let mut stream = sequence.create_execution_stream().unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _selector = sequence.authority_source.lock().unwrap();
        panic!("injected authority-source selector panic");
    }));
    assert!(panic.is_err());
    assert!(sequence.is_poisoned());
    let source = match sequence.authority_source.lock() {
        Ok(source) => *source,
        Err(poisoned) => *poisoned.into_inner(),
    };
    assert_eq!(source, SequenceExecutionAuthoritySource::FailClosed);
    assert!(sequence.open_session().is_err());
    assert!(sequence.activate(&mut stream).is_err());

    drop(stream);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_session_live_witness_accepts_only_the_exact_open_identity() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "live-witness-open");
    let session = sequence.open_session().unwrap();
    let slot_strong_count = Arc::strong_count(&session.slot);
    let witness = session.live_witness().unwrap();

    assert_eq!(Arc::strong_count(&session.slot), slot_strong_count);
    witness.ensure_open().unwrap();
    witness.ensure_live().unwrap();
    witness
        .ensure_identity(session.epoch(), session.fingerprint())
        .unwrap();
    witness
        .ensure_live_identity(session.epoch(), session.fingerprint())
        .unwrap();
    let other_epoch = SequenceSessionEpoch(
        NonZeroU64::new(session.epoch().get() + 1).expect("the next test epoch is non-zero"),
    );
    assert!(witness
        .ensure_identity(other_epoch, session.fingerprint())
        .is_err());
    assert!(witness
        .ensure_identity(
            session.epoch(),
            &SequenceSessionFingerprint("different-session".to_owned()),
        )
        .is_err());

    session.request_cancel().unwrap();
    assert!(witness.ensure_open().is_err());
    assert!(witness
        .ensure_identity(session.epoch(), session.fingerprint())
        .is_err());
    witness.ensure_live().unwrap();
    witness
        .ensure_live_identity(session.epoch(), session.fingerprint())
        .unwrap();
    assert!(witness
        .ensure_live_identity(other_epoch, session.fingerprint())
        .is_err());
    let terminal = session.try_abort().unwrap();
    assert!(witness.ensure_live().is_err());
    drop(terminal);
    drop(witness);
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_session_live_witness_rejects_terminal_and_dropped_sessions() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'c',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let _initialization = harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();

    let completed_sequence = admitted_sequence(&harness.root, "live-witness-completed");
    let completed_session = completed_sequence.open_session().unwrap();
    let completed_witness = completed_session.live_witness().unwrap();
    {
        let mut state = completed_session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            panic!("newly opened test session must be active");
        };
        active.retired_frames = 1;
    }
    let completed = completed_session.try_complete().unwrap();
    assert_eq!(
        completed.disposition(),
        SequenceSessionTerminalDisposition::Completed
    );
    assert!(completed_witness.ensure_open().is_err());
    drop(completed);
    drop(completed_witness);
    drop(completed_session);
    drop(completed_sequence);

    let aborted_sequence = admitted_sequence(&harness.root, "live-witness-aborted");
    let aborted_session = aborted_sequence.open_session().unwrap();
    let aborted_witness = aborted_session.live_witness().unwrap();
    aborted_session.request_cancel().unwrap();
    let aborted = aborted_session.try_abort().unwrap();
    assert_eq!(
        aborted.disposition(),
        SequenceSessionTerminalDisposition::Aborted
    );
    assert!(aborted_witness.ensure_open().is_err());
    drop(aborted);
    drop(aborted_witness);
    drop(aborted_session);
    drop(aborted_sequence);

    let dropped_sequence = admitted_sequence(&harness.root, "live-witness-dropped");
    let dropped_session = dropped_sequence.open_session().unwrap();
    let dropped_witness = dropped_session.live_witness().unwrap();
    drop(dropped_session);
    assert!(dropped_sequence.is_poisoned());
    assert!(dropped_witness.ensure_open().is_err());
    drop(dropped_sequence);
    assert!(dropped_witness.slot.upgrade().is_none());
    assert!(dropped_witness.ensure_open().is_err());
    drop(dropped_witness);

    close_dynamic_test_root(harness.root);
}

#[test]
fn multi_pool_batch_publishes_one_capacity_epoch() {
    let combined = combine_catalogs(&[
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'a',
            1,
            256,
            TestDemand::Fixed,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'b',
            1,
            256,
            TestDemand::Fixed,
        ),
    ]);
    let runtime = new_runtime(&combined, 512);
    let harness = harness(runtime, combined, 512, false);
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    let receipt = harness
        .root
        .maintenance_controller
        .grow_pools(
            harness
                .pool_ids
                .iter()
                .rev()
                .cloned()
                .map(|pool_id| DynamicPoolGrowthRequest::new(pool_id, 64).unwrap())
                .collect(),
        )
        .unwrap();
    assert_eq!(receipt.growths().len(), 2);
    assert_eq!(receipt.capacity_epoch(), before.capacity_epoch() + 1);
    assert!(receipt
        .growths()
        .iter()
        .all(|growth| growth.capacity_epoch() == receipt.capacity_epoch()));
    let after = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(after
        .domains()
        .iter()
        .all(|domain| domain.total().get() == 64));
    assert_eq!(harness.runtime.allocate_calls(), 2);
}

#[test]
fn multi_pool_nth_allocation_failure_has_zero_partial_publication() {
    let combined = combine_catalogs(&[
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'a',
            1,
            256,
            TestDemand::Fixed,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'b',
            1,
            256,
            TestDemand::Fixed,
        ),
    ]);
    let runtime = new_runtime(&combined, 512);
    runtime.fail_on_call(2);
    let harness = harness(Arc::clone(&runtime), combined, 512, false);
    let requests = harness
        .pool_ids
        .iter()
        .cloned()
        .map(|pool_id| DynamicPoolGrowthRequest::new(pool_id, 64).unwrap())
        .collect::<Vec<_>>();
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(harness
        .root
        .maintenance_controller
        .grow_pools(requests.clone())
        .is_err());
    let after = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(after.capacity_epoch(), before.capacity_epoch());
    assert!(after
        .domains()
        .iter()
        .all(|domain| domain.total() == CapacityUnits::ZERO));
    for pool in harness.root.dynamic_pools.pools.values() {
        let state = pool.state.lock().unwrap();
        assert_eq!(state.pending_growth_bytes, 0);
        assert!(state.chunks.is_empty());
    }
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .budget
            .account
            .state
            .lock()
            .unwrap()
            .claimed_bytes,
        0
    );
    runtime.fail_on_call(0);
    assert_eq!(
        harness
            .root
            .maintenance_controller
            .grow_pools(requests)
            .unwrap()
            .growths()
            .len(),
        2
    );
}

#[test]
fn publication_failure_withdraws_chunks_to_typed_quarantine() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, true);
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .is_err());
    let after = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(after.capacity_epoch(), before.capacity_epoch());
    assert_eq!(after.domains()[0].total(), before.domains()[0].total());
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let state = pool.state.lock().unwrap();
    assert!(state.chunks.is_empty());
    assert_eq!(state.quarantined.len(), 1);
    assert_eq!(
        state.quarantined[0].reason,
        DynamicChunkQuarantineReason::PublicationRejected
    );
    assert_eq!(state.quarantined[0].backing.descriptor.size_bytes, 64);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .budget
            .account
            .state
            .lock()
            .unwrap()
            .claimed_bytes,
        64
    );
}

#[test]
fn scoped_demand_merges_per_pool_and_release_coalesces_extents() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        2,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let (demand, requested) = binding
        .scoped_demand(
            AllocationLifetime::Request,
            None,
            shape(1),
            shape(1),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    assert_eq!(demand.immediate_claim().entries().len(), 1);
    assert_eq!(demand.immediate_claim().entries()[0].units().get(), 128);
    assert_eq!(requested.len(), 2);
    let admitted = match binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/pool-merge").unwrap(),
            RequestIdentity::new("request/pool-merge").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::Admitted(admitted) => admitted,
        _ => panic!("initialized merged pool must admit"),
    };
    assert_eq!(admitted.backing_slices().len(), 2);
    drop(admitted);
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let state = pool.state.lock().unwrap();
    let chunk = state.chunks.values().next().unwrap();
    assert_eq!(state.allocator.free_bytes, 128);
    assert_eq!(state.allocator.largest_contiguous_bytes(), 128);
    assert_eq!(state.allocator.by_offset.len(), 1);
    assert_eq!(
        state
            .allocator
            .by_offset
            .values()
            .next()
            .unwrap()
            .length_bytes,
        128
    );
    assert_eq!(chunk.live_segments, 0);
}

#[test]
fn request_token_backing_covers_nonzero_chunk_source_range() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let request = RequestResourceAdmissionRequest::new(
        chunked_work(4, 2..3),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let admitted = match binding
        .try_admit_request(
            request,
            RunId::new("run/chunked-request-backing").unwrap(),
            RequestIdentity::new("request/chunked-request-backing").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::Admitted(admitted) => admitted,
        _ => panic!("resident full-input request backing must admit"),
    };

    assert_eq!(admitted.work_shape().immediate_tokens(), 1);
    assert_eq!(admitted.work_shape().fit_tokens(), 4);
    assert_eq!(admitted.backing_slices().len(), 1);
    assert_eq!(admitted.backing_slices()[0].size_bytes(), 256);
    let snapshot = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(snapshot.domains()[0].used().get(), 256);

    drop(admitted);
    drop(binding);
    close_dynamic_test_root(harness.root);
}

#[test]
fn step_captures_the_exact_sequence_backing_snapshot() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "sequence-backing-snapshot");
    let admitted_snapshot = sequence.backing_snapshot().unwrap();
    assert_eq!(admitted_snapshot.generation().get(), 1);
    assert_eq!(admitted_snapshot.backing_slices().len(), 1);
    assert_eq!(admitted_snapshot.backing_slices()[0].size_bytes(), 64);

    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(1)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = match batch.try_begin_step(request).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident sequence snapshot step must admit"),
    };
    assert!(Arc::ptr_eq(
        step.participant_backing_snapshot(BatchParticipantAuthority::new(
            sequence.sequence_authority(),
            sequence.request_authority(),
        ))
        .unwrap(),
        &admitted_snapshot
    ));

    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(admitted_snapshot);
    close_dynamic_test_root(harness.root);
}

#[test]
fn step_backing_deferral_retains_exact_participant_session_until_retry() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    let sequence = admitted_sequence(&harness.root, "step-deferral-parent");
    let session = sequence.open_session().unwrap();
    let session_weak = Arc::downgrade(&session);
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(1)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();

    let deferred = match batch.try_begin_step(request.clone()).unwrap() {
        StepResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident step backing must defer"),
    };
    assert_eq!(deferred.participant_count(), 1);
    assert_eq!(
        deferred.work_fingerprint(),
        request.work_shape().fingerprint()
    );
    drop(batch);
    drop(session);
    assert!(session_weak.upgrade().is_some());

    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    let retained_session = session_weak
        .upgrade()
        .expect("step backing authority retains its exact participant session");
    drop(deferred);
    let retry_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&retained_session)]).unwrap();
    let step = match retry_batch.try_begin_step(request).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("maintained step backing must admit for the retained participant"),
    };
    step.try_retire_normal().unwrap();
    retained_session.try_complete().unwrap();
    drop(retry_batch);
    drop(retained_session);
    assert!(session_weak.upgrade().is_none());
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_backing_extension_publishes_atomically_between_frames() {
    let catalog = pool_catalog_with_options(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
        "state",
        false,
        StateInitialization::Zero,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "sequence-backing-extension", 2);
    let initial = sequence.backing_snapshot().unwrap();
    let initial_cell = Arc::clone(initial.backing_slices()[0].initialization_cell().unwrap());
    assert!(initial_cell.prepare("wave/initial").unwrap());
    initial_cell.mark_in_flight("wave/initial").unwrap();
    initial_cell.finish("wave/initial", true).unwrap();
    let session = sequence.open_session().unwrap();
    let active_before = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let participant =
        BatchParticipantAuthority::new(sequence.sequence_authority(), sequence.request_authority());

    let first_step = match batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(1)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident first frame must admit"),
    };
    let first_captured = Arc::clone(
        first_step
            .participant_backing_snapshot(participant)
            .unwrap(),
    );
    match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::RetryRequired(current) => {
            assert!(Arc::ptr_eq(&current, &initial));
        }
        _ => panic!("an active frame must defer sequence backing publication"),
    }
    first_step.try_retire_normal().unwrap();

    let extended = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("resident paged capacity must extend after frame retirement"),
    };
    assert_eq!(initial.generation().get(), 1);
    assert_eq!(first_captured.generation().get(), 1);
    assert_eq!(extended.generation().get(), 2);
    assert_eq!(extended.committed_tokens(), 2);
    assert_eq!(extended.backing_slices().len(), 2);
    assert!(Arc::ptr_eq(
        extended.backing_slices()[0].initialization_cell().unwrap(),
        &initial_cell
    ));
    assert_eq!(
        extended.backing_slices()[0]
            .initialization_status()
            .unwrap(),
        Some(BackingInitializationStatus::Initialized)
    );
    assert!(!Arc::ptr_eq(
        extended.backing_slices()[1].initialization_cell().unwrap(),
        &initial_cell
    ));
    assert_eq!(
        extended.backing_slices()[1]
            .initialization_status()
            .unwrap(),
        Some(BackingInitializationStatus::Pending)
    );
    assert_eq!(
        extended
            .backing_slices()
            .iter()
            .map(LogicalBackingSliceAuthority::size_bytes)
            .sum::<u64>(),
        128
    );
    let active_after = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    assert_eq!(active_before.fingerprint(), active_after.fingerprint());

    let second_step = match batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(1)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident second frame must admit"),
    };
    assert!(Arc::ptr_eq(
        second_step
            .participant_backing_snapshot(participant)
            .unwrap(),
        &extended
    ));
    let resource_id = extended.backing_slices()[0].resource_id().clone();
    let view = second_step
        .participant_backing_view(participant, &resource_id)
        .unwrap();
    assert_eq!(view.size_bytes(), 128);
    assert_eq!(view.segment_bindings().len(), 2);

    second_step.try_retire_normal().unwrap();
    let covered_prefix = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(1), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Current(snapshot) => snapshot,
        _ => panic!("a committed backing frontier must cover a narrower execution prefix"),
    };
    assert!(Arc::ptr_eq(&covered_prefix, &extended));

    let over_ceiling = match session.try_ensure_backing_covers(
        SequenceResourceExtensionRequest::new(work(3), AdmissionPressureAction::WaitForRelease)
            .unwrap(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("sequence extension above the request ceiling must fail"),
    };
    assert!(over_ceiling
        .to_string()
        .contains("parent request token ceiling"));
    session.try_complete().unwrap();
    drop(active_after);
    drop(active_before);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(first_captured);
    drop(initial);
    drop(covered_prefix);
    drop(extended);
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(!released.poisoned());
    assert_eq!(released.active_child_claims(), 0);
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    assert!(released
        .domains()
        .iter()
        .all(|domain| domain.used().get() == 0));
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_backing_extension_waits_for_released_capacity_then_retries() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 128)
        .unwrap();
    let first = admitted_sequence_with_ceiling(&harness.root, "extension-wait-first", 2);
    let second = admitted_sequence_with_ceiling(&harness.root, "extension-wait-second", 2);
    let session = first.open_session().unwrap();

    let deferred = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("resident sequence backing pressure must wait for release"),
    };
    let waiter = deferred.register_waiter().unwrap();
    assert!(!waiter.recheck().unwrap().should_retry());

    drop(second);
    assert!(waiter.recheck().unwrap().should_retry());
    drop(waiter);
    drop(deferred);

    let extended = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("released backing capacity must admit the pending extension"),
    };
    assert_eq!(extended.generation().get(), 2);
    assert_eq!(extended.committed_tokens(), 2);
    assert_eq!(extended.backing_slices().len(), 2);

    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop(session);
    drop(first);
    drop(extended);
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(!released.poisoned());
    assert_eq!(released.active_child_claims(), 0);
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    assert!(released
        .domains()
        .iter()
        .all(|domain| domain.used().get() == 0));
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_extension_deferral_retains_exact_session_and_rejects_stale_generation() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 128)
        .unwrap();
    let first = admitted_sequence_with_ceiling(&harness.root, "extension-owner-first", 2);
    let second = admitted_sequence_with_ceiling(&harness.root, "extension-owner-second", 2);
    let session = first.open_session().unwrap();
    let target = work(2);

    let deferred = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(
                target.clone(),
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("occupied backing must defer the exact sequence extension"),
    };
    assert_eq!(deferred.expected_generation().get(), 1);
    assert_eq!(deferred.target_fingerprint(), target.fingerprint());

    let session_weak = Arc::downgrade(&session);
    drop(session);
    assert!(session_weak.upgrade().is_some());
    drop(second);

    let retained_session = session_weak
        .upgrade()
        .expect("the deferred authority retains its exact session");
    let extended = match retained_session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(target, AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("released backing must extend through the retained session"),
    };
    assert_eq!(extended.generation().get(), 2);
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
    ));

    retained_session.request_cancel().unwrap();
    let terminal = retained_session.try_abort().unwrap();
    assert!(deferred.maintain().is_err());
    drop(terminal);
    drop(deferred);
    drop(retained_session);
    assert!(session_weak.upgrade().is_none());
    drop(first);
    drop(extended);
    close_dynamic_test_root(harness.root);
}

#[test]
fn invocation_subset_maps_its_local_participant_to_the_step_snapshot() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness_with_nodes(
        runtime,
        catalog,
        256,
        false,
        Arc::from(vec![PlanNode::resource_test_node(
            NodeId::new("node/dynamic-pool-test").unwrap(),
        )]),
    );
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let first = admitted_sequence(&harness.root, "snapshot-subset-first");
    let second = admitted_sequence(&harness.root, "snapshot-subset-second");
    let first_session = first.open_session().unwrap();
    let second_session = second.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![
        Arc::clone(&second_session),
        Arc::clone(&first_session),
    ])
    .unwrap();
    let step_request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![token_span(1), token_span(1)])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = match batch.try_begin_step(step_request).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident two-participant step must admit"),
    };

    let selected = batch.sessions()[1].resources();
    let selected_snapshot = selected.backing_snapshot().unwrap();
    let unselected_snapshot = batch.sessions()[0].resources().backing_snapshot().unwrap();
    let selected_authority =
        BatchParticipantAuthority::new(selected.sequence_authority(), selected.request_authority());
    let invocation_request = InvocationResourceAdmissionRequest::new(
        NodeId::new("node/dynamic-pool-test").unwrap(),
        step.bind_invocation_work_shape(vec![(selected_authority, token_span(1))])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let invocation = match step.try_admit_invocation(invocation_request).unwrap() {
        InvocationResourceAdmissionDecision::Admitted(invocation) => invocation,
        _ => panic!("resident invocation subset must admit"),
    };
    let captured = invocation.participant_backing_snapshot(0).unwrap();
    assert!(Arc::ptr_eq(captured, &selected_snapshot));
    assert!(!Arc::ptr_eq(captured, &unselected_snapshot));

    drop(invocation);
    step.try_retire_normal().unwrap();
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop(batch);
    drop(first_session);
    drop(second_session);
    drop(first);
    drop(second);
    drop(selected_snapshot);
    drop(unselected_snapshot);
    close_dynamic_test_root(harness.root);
}

#[test]
fn step_slot_projects_two_logical_activations_from_one_physical_extent() {
    let catalog = shared_step_activation_catalog(linear_profile());
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence(&harness.root, "shared-step-slot");
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(1)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = match batch.try_begin_step(request).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident shared Step slot must admit"),
    };

    assert_eq!(
        step.claimed_backing().demand().immediate_claim().entries()[0]
            .units()
            .get(),
        64
    );
    assert_eq!(step.backing_slices().len(), 2);
    assert_eq!(step.claimed_backing().physical_claim_count(), 1);
    assert!(step.claimed_backing().has_shared_physical_claims());
    let first = step.backing_slices()[0].evidence();
    let second = step.backing_slices()[1].evidence();
    assert_eq!(
        first.physical_claim_identity(),
        second.physical_claim_identity()
    );
    assert_eq!(first.physical_size_bytes(), 64);
    assert_eq!(second.physical_size_bytes(), 64);
    assert_eq!(first.segments(), second.segments());
    assert_eq!(
        harness
            .root
            .maintenance_controller
            .status()
            .unwrap()
            .pools()[0]
            .live_segments(),
        1
    );
    let invocation = InvocationResourceAdmissionRequest::for_all_step_participants(
        NodeId::new("node/shared-step-slot").unwrap(),
        step.bind_all_invocation_work_shape(vec![token_span(1)])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let error = match step.try_admit_invocation(invocation) {
        Err(error) => error,
        Ok(_) => panic!("single-node dispatch consumed a wave-only Step slot"),
    };
    assert!(error.to_string().contains("single-fence submission wave"));

    step.try_retire_normal().unwrap();
    let status = harness.root.maintenance_controller.status().unwrap();
    assert_eq!(status.pools()[0].live_segments(), 0);
    assert_eq!(status.pools()[0].free_bytes(), 64);
}

#[test]
fn contiguous_profile_rejects_fragmented_cross_chunk_claim() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&harness.pool_ids[0]).unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 128).unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let first = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let middle = claim_size(&harness.root.dynamic_pools, &pool, 64);
    let last = claim_size(&harness.root.dynamic_pools, &pool, 64);
    drop(first);
    drop(last);
    let request = evaluated_request(
        &pool,
        pool.domain.descriptors[0]
            .evaluate_request_bytes(&work(2))
            .unwrap(),
    );
    let BackingPrepareDecision::Deferred(deferred) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("contiguous claim across fragmented chunks must defer")
    };
    assert_eq!(deferred.blockers().len(), 1);
    assert_eq!(
        deferred.blockers()[0].reason(),
        DynamicBackingDeferralReason::FragmentedContiguous
    );
    assert_eq!(deferred.blockers()[0].free_bytes(), 128);
    assert_eq!(deferred.blockers()[0].largest_contiguous_bytes(), 64);
    let registration = binding.register_backing_waiter(&deferred).unwrap();
    assert!(!registration.recheck().unwrap().should_retry());
    drop(middle);
    assert!(registration.recheck().unwrap().should_retry());
    let BackingPrepareDecision::Prepared(prepared) = harness
        .root
        .dynamic_pools
        .prepare_claim(std::slice::from_ref(&request))
        .unwrap()
    else {
        panic!("coalesced contiguous claim must prepare")
    };
    let authority = prepared.commit().pop().unwrap();
    assert_eq!(authority.evidence().segments().len(), 1);
    assert_eq!(authority.evidence().segments()[0].length_bytes(), 128);
}

#[test]
fn contiguous_batch_deferral_retains_whole_packing_demand_and_grows_to_progress() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'c',
        2,
        2_048,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 2_048);
    let harness = harness(runtime, catalog, 2_048, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.grow_pool(&harness.pool_ids[0], 640).unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 256).unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let requests = vec![
        evaluated_descriptor_request(&pool, 0, 448),
        evaluated_descriptor_request(&pool, 1, 448),
    ];

    let BackingPrepareDecision::Deferred(deferred) =
        harness.root.dynamic_pools.prepare_claim(&requests).unwrap()
    else {
        panic!("640+256 free bytes cannot initially pack two 448-byte claims")
    };
    let blocker = &deferred.blockers()[0];
    assert_eq!(
        blocker.reason(),
        DynamicBackingDeferralReason::FragmentedContiguous
    );
    assert_eq!(blocker.free_bytes(), 896);
    assert_eq!(blocker.largest_contiguous_bytes(), 640);
    assert_eq!(blocker.requested_bytes(), 448);
    assert_eq!(
        blocker.contiguous_claim_bytes_descending(),
        Some([448, 448].as_slice())
    );
    assert!(blocker
        .free_extent_layout_fingerprint()
        .starts_with("sha256/"));

    let DynamicDeferredMaintenanceOutcome::Maintained(growth) =
        maintenance.maintain_for_live_deferred(&deferred).unwrap()
    else {
        panic!("whole-transaction fragmentation must grow or wait, never retry unchanged")
    };
    assert_eq!(growth.growths().len(), 1);
    assert_eq!(growth.growths()[0].chunk_bytes(), 448);

    let BackingPrepareDecision::Prepared(prepared) =
        harness.root.dynamic_pools.prepare_claim(&requests).unwrap()
    else {
        panic!("the progress-producing growth must make the exact batch packable")
    };
    let authorities = prepared.commit();
    assert_eq!(authorities.len(), 2);
    drop(authorities);
    let status = maintenance.status().unwrap();
    assert_eq!(status.pools()[0].resident_bytes(), 1_344);
    assert_eq!(status.pools()[0].free_bytes(), 1_344);
}

#[test]
fn contiguous_batch_shortfall_grows_for_packability_not_aggregate_bytes() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'd',
        3,
        4_096,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 4_096);
    let harness = harness(runtime, catalog, 4_096, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.grow_pool(&harness.pool_ids[0], 368).unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let requests = vec![
        evaluated_descriptor_request(&pool, 0, 720),
        evaluated_descriptor_request(&pool, 1, 720),
        evaluated_descriptor_request(&pool, 2, 720),
    ];

    let BackingPrepareDecision::Deferred(deferred) =
        harness.root.dynamic_pools.prepare_claim(&requests).unwrap()
    else {
        panic!("one undersized chunk cannot hold three contiguous claims")
    };
    let blocker = &deferred.blockers()[0];
    assert_eq!(
        blocker.reason(),
        DynamicBackingDeferralReason::GrowthRequired
    );
    assert_eq!(blocker.free_bytes(), 368);
    assert_eq!(blocker.largest_contiguous_bytes(), 368);
    assert_eq!(blocker.requested_bytes(), 2_160);
    assert_ne!(blocker.requested_bytes(), 2_160 - 368);

    let DynamicDeferredMaintenanceOutcome::Maintained(growth) =
        maintenance.maintain_for_live_deferred(&deferred).unwrap()
    else {
        panic!("contiguous shortfall must install one transaction-packable chunk")
    };
    assert_eq!(growth.growths()[0].chunk_bytes(), 2_160);
    let BackingPrepareDecision::Prepared(prepared) =
        harness.root.dynamic_pools.prepare_claim(&requests).unwrap()
    else {
        panic!("one transaction-packable growth must satisfy all three claims")
    };
    let authorities = prepared.commit();
    assert_eq!(authorities.len(), 3);
    drop(authorities);
}

#[test]
fn paged_profile_claims_across_resident_chunks() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&harness.pool_ids[0]).unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 64).unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let authority = claim_size(&harness.root.dynamic_pools, &pool, 128);
    assert_eq!(authority.evidence().segments().len(), 2);
    assert_ne!(
        authority.evidence().segments()[0].chunk_ordinal(),
        authority.evidence().segments()[1].chunk_ordinal()
    );
    assert_eq!(
        authority
            .evidence()
            .segments()
            .iter()
            .map(|segment| segment.length_bytes())
            .sum::<u64>(),
        128
    );
}

#[test]
fn two_plans_contend_on_one_process_wide_device_account() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let first = harness(Arc::clone(&runtime), catalog.clone(), 128, false);
    let second = harness(Arc::clone(&runtime), catalog, 96, false);
    let foreign = match first
        .root
        .trusted_runtime_binding()
        .unwrap()
        .try_admit_request(
            request_admission(),
            RunId::new("run/foreign-deferred").unwrap(),
            RequestIdentity::new("request/foreign-deferred").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident first plan must defer"),
    };
    assert!(
        PlanBackingDeferral::new(Arc::clone(&second.root), foreign.evidence().clone(),).is_err()
    );
    let first_status = &first.root.maintenance_controller.status().unwrap();
    let second_status = &second.root.maintenance_controller.status().unwrap();
    assert_eq!(first_status.effective_device_usable_ceiling_bytes(), 96);
    assert_eq!(second_status.effective_device_usable_ceiling_bytes(), 96);
    assert_eq!(first_status.budget_device_wide_usable_ceiling_bytes(), 128);
    assert_eq!(second_status.budget_device_wide_usable_ceiling_bytes(), 96);
    first
        .root
        .maintenance_controller
        .grow_pool(&first.pool_ids[0], 96)
        .unwrap();
    let first_status = &first.root.maintenance_controller.status().unwrap();
    let second_status = &second.root.maintenance_controller.status().unwrap();
    assert_eq!(first_status.process_claimed_bytes(), 96);
    assert_eq!(first_status.budget_claimed_bytes(), 96);
    assert_eq!(second_status.budget_claimed_bytes(), 0);
    let second_deferred = match second
        .root
        .trusted_runtime_binding()
        .unwrap()
        .try_admit_request(
            request_admission(),
            RunId::new("run/process-wide-deferred").unwrap(),
            RequestIdentity::new("request/process-wide-deferred").unwrap(),
        )
        .unwrap()
    {
        RequestResourceAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident second plan must defer its own backing"),
    };
    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        wait_condition,
        pressure,
        ..
    } = second_deferred.maintain().unwrap()
    else {
        panic!("process-wide pressure must become a typed capacity wait")
    };
    assert_eq!(pressure.scope(), &DeviceCapacityPressureScope::ProcessWide);
    assert_eq!(pressure.requested_bytes(), 64);
    assert_eq!(pressure.available_bytes(), 0);
    assert!(wait_condition
        .observed()
        .iter()
        .any(|entry| { entry.source() == CapacityAvailabilitySource::ProcessDeviceCapacity }));
    let registration = second
        .root
        .register_capacity_waiter(&wait_condition)
        .unwrap();
    assert!(!registration.recheck().unwrap().should_retry());
    assert!(matches!(
        second
            .root
            .maintenance_controller
            .grow_pool(&second.pool_ids[0], 64),
        Err(VNextError::DeviceCapacityUnavailable(pressure))
            if pressure.scope() == &DeviceCapacityPressureScope::ProcessWide
                && pressure.requested_bytes() == 64
                && pressure.available_bytes() == 0
    ));
    drop(foreign);
    drop(first);
    assert!(registration.recheck().unwrap().should_retry());
    second
        .root
        .maintenance_controller
        .grow_pool(&second.pool_ids[0], 64)
        .unwrap();
}

#[test]
fn ten_thousand_steady_claim_release_cycles_allocate_zero_device_buffers() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        64,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 64);
    let harness = harness(Arc::clone(&runtime), catalog, 64, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let baseline_allocations = runtime.allocate_calls();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    for _ in 0..10_000 {
        drop(claim_size(&harness.root.dynamic_pools, &pool, 64));
    }
    assert_eq!(runtime.allocate_calls() - baseline_allocations, 0);
    let state = pool.state.lock().unwrap();
    assert_eq!(state.allocator.free_bytes, 64);
    assert_eq!(state.allocator.largest_contiguous_bytes(), 64);
    assert_eq!(state.allocator.by_offset.len(), 1);
    assert_eq!(
        state
            .allocator
            .by_offset
            .values()
            .next()
            .unwrap()
            .length_bytes,
        64
    );
}

#[test]
fn controller_maintains_current_deferral_and_retries_stale_epoch_without_growth() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(Arc::clone(&runtime), catalog, 256, false);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let decision = binding
        .try_admit_request(
            request_admission(),
            RunId::new("run/controller-deferred").unwrap(),
            RequestIdentity::new("request/controller-deferred").unwrap(),
        )
        .unwrap();
    let RequestResourceAdmissionDecision::BackingDeferred(deferred) = decision else {
        panic!("zero-resident pool must expose physical deferral")
    };
    let before = &harness.root.maintenance_controller.status().unwrap();
    assert_eq!(before.pools().len(), 1);
    assert_eq!(before.pools()[0].pool_id(), &harness.pool_ids[0]);
    assert_eq!(
        before.pools()[0].domain_id(),
        harness.root.dynamic_pools.pools[&harness.pool_ids[0]]
            .domain
            .domain_id
    );
    assert_eq!(before.pools()[0].resident_bytes(), 0);
    assert_eq!(before.budget_claimed_bytes(), 0);

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = deferred.maintain().unwrap()
    else {
        panic!("current deferral must be maintained")
    };
    assert_eq!(receipt.growths().len(), 1);
    assert_eq!(runtime.allocate_calls(), 1);
    let DynamicDeferredMaintenanceOutcome::RetryAdmission { current_epochs } =
        deferred.maintain().unwrap()
    else {
        panic!("stale deferral must retry without another growth")
    };
    assert_ne!(current_epochs, deferred.evidence().epochs());
    assert_eq!(runtime.allocate_calls(), 1);
}

#[test]
fn sequence_backing_deferral_remains_current_while_parent_request_is_retained() {
    let request_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'd',
        1,
        64,
        TestDemand::Fixed,
    );
    let sequence_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'e',
        1,
        128,
        TestDemand::Fixed,
    );
    let request_pool_id = request_catalog.pool_id.clone();
    let sequence_pool_id = sequence_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[request_catalog, sequence_catalog]);
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(Arc::clone(&runtime), catalog, 192, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&request_pool_id)
        .unwrap();
    let request = admitted_request(&harness.root, "retained-parent-deferral");
    let admission = SequenceResourceAdmissionRequest::new(
        work(1),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let SequenceResourceAdmissionDecision::BackingDeferred(deferred) =
        request.try_admit_sequence(admission).unwrap()
    else {
        panic!("zero-resident sequence backing must defer")
    };

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = deferred.maintain().unwrap()
    else {
        panic!("retained parent must keep the child backing evidence current")
    };
    assert_eq!(receipt.growths().len(), 1);
    assert_eq!(receipt.growths()[0].pool_id(), &sequence_pool_id);
    assert_eq!(runtime.allocate_calls(), 2);

    drop(deferred);
    drop(request);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_backing_deferral_retains_exact_parent_after_external_parent_release() {
    let request_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'f',
        1,
        64,
        TestDemand::Fixed,
    );
    let sequence_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        '1',
        1,
        128,
        TestDemand::Fixed,
    );
    let request_pool_id = request_catalog.pool_id.clone();
    let catalog = combine_catalogs(&[request_catalog, sequence_catalog]);
    let runtime = new_runtime(&catalog, 192);
    let harness = harness(Arc::clone(&runtime), catalog, 192, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&request_pool_id)
        .unwrap();
    let request = admitted_request(&harness.root, "released-parent-deferral");
    let admission = SequenceResourceAdmissionRequest::new(
        work(1),
        AdmissionFitPolicy::FullInputMustFit,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let SequenceResourceAdmissionDecision::BackingDeferred(deferred) =
        request.try_admit_sequence(admission).unwrap()
    else {
        panic!("zero-resident sequence backing must defer")
    };
    let parent_authority = request.request_authority();
    assert_eq!(deferred.parent().request_authority(), parent_authority);
    drop(request);

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = deferred.maintain().unwrap()
    else {
        panic!("typed sequence deferral must retain its exact parent authority")
    };
    assert_eq!(receipt.growths().len(), 1);
    assert_eq!(runtime.allocate_calls(), 2);
    let retained = deferred.into_parent();
    assert_eq!(retained.request_authority(), parent_authority);
    drop(retained);
    close_dynamic_test_root(harness.root);
}

#[test]
fn descriptor_mismatch_is_observable_and_explicit_cleanup_returns_its_grant() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    runtime.mismatch_on_call(1);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    assert!(harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .is_err());
    let quarantined = &harness.root.maintenance_controller.status().unwrap();
    assert_eq!(quarantined.process_claimed_bytes(), 64);
    assert_eq!(quarantined.budget_claimed_bytes(), 64);
    assert_eq!(quarantined.pools()[0].resident_bytes(), 0);
    assert_eq!(quarantined.pools()[0].quarantined_chunks(), 1);
    assert_eq!(quarantined.pools()[0].quarantined_bytes(), 64);
    assert_eq!(quarantined.pools()[0].descriptor_mismatch_chunks(), 1);
    assert_eq!(quarantined.pools()[0].publication_rejected_chunks(), 0);
    let before_release = harness
        .root
        .dynamic_pools
        .budget
        .availability_snapshot()
        .unwrap();

    let released = harness
        .root
        .maintenance_controller
        .release_quarantined_chunks()
        .unwrap();
    assert_eq!(released.released_chunks(), 1);
    assert_eq!(released.released_bytes(), 64);
    assert_eq!(released.pools().len(), 1);
    let clean = &harness.root.maintenance_controller.status().unwrap();
    assert_eq!(clean.process_claimed_bytes(), 0);
    assert_eq!(clean.budget_claimed_bytes(), 0);
    assert_eq!(clean.pools()[0].quarantined_chunks(), 0);
    let after_release = harness
        .root
        .dynamic_pools
        .budget
        .availability_snapshot()
        .unwrap();
    assert!(after_release.plan_epoch() > before_release.plan_epoch());
    assert!(after_release.process_epoch() > before_release.process_epoch());
}

#[test]
fn concurrent_initialization_allocates_and_publishes_exactly_once() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    let barrier = Arc::new(std::sync::Barrier::new(DYNAMIC_POOL_CONCURRENT_WORKERS + 1));
    let outcomes = std::thread::scope(|scope| {
        let worker_barrier = Arc::clone(&barrier);
        let controller = &harness.root.maintenance_controller;
        let worker_pool = harness.pool_ids[0].clone();
        let caller_pool = harness.pool_ids[0].clone();
        let worker = std::thread::Builder::new()
            .name("vnext-dynamic-pool-initializer".to_owned())
            .spawn_scoped(scope, move || {
                worker_barrier.wait();
                controller.initialize_pool(&worker_pool).unwrap().is_some()
            })
            .expect("the single bounded dynamic-pool worker starts");
        barrier.wait();
        let caller = controller.initialize_pool(&caller_pool).unwrap().is_some();
        [caller, worker.join().unwrap()]
    });
    assert_eq!(outcomes.into_iter().filter(|grew| *grew).count(), 1);
    assert_eq!(runtime.allocate_calls(), 1);
    let status = &harness.root.maintenance_controller.status().unwrap();
    assert_eq!(status.pools()[0].resident_bytes(), 64);
    assert_eq!(status.pools()[0].resident_chunks(), 1);
}

#[test]
fn maintenance_panic_cancels_pending_capacity_and_poison_closes_the_controller() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    runtime.panic_on_call(1);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = harness
            .root
            .maintenance_controller
            .initialize_pool(&harness.pool_ids[0]);
    }));
    assert!(result.is_err());
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let state = pool.state.lock().unwrap();
    assert_eq!(state.pending_growth_bytes, 0);
    assert_eq!(state.resident_bytes, 0);
    assert!(state.chunks.is_empty());
    drop(state);
    let account = harness
        .root
        .dynamic_pools
        .budget
        .account
        .state
        .lock()
        .unwrap();
    assert_eq!(account.claimed_bytes, 0);
    drop(account);
    assert!(harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .is_err());
}

#[test]
fn unsplit_plan_owner_drops_all_buffers_before_backend_context() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 128);
    let dropped_after_backend = runtime.dropped_after_backend_probe();
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let Harness {
        root,
        runtime: harness_runtime,
        pool_ids: _,
    } = harness;
    let pools = Arc::downgrade(&root.dynamic_pools);
    drop(harness_runtime);
    drop(runtime);
    drop(root);
    assert!(pools.upgrade().is_none());
    assert!(!dropped_after_backend.load(Ordering::Acquire));
}

#[test]
fn multi_pool_invalid_request_and_prepared_drop_have_zero_partial_claim() {
    let combined = combine_catalogs(&[
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'a',
            1,
            128,
            TestDemand::Fixed,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'b',
            1,
            128,
            TestDemand::Fixed,
        ),
    ]);
    let runtime = new_runtime(&combined, 256);
    let harness = harness(runtime, combined, 256, false);
    harness
        .root
        .maintenance_controller
        .initialize_pools(&harness.pool_ids)
        .unwrap();
    let first = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let second = &harness.root.dynamic_pools.pools[&harness.pool_ids[1]];
    let valid = evaluated_request(first, 64);
    let invalid = evaluated_request(second, 1);
    assert!(harness
        .root
        .dynamic_pools
        .prepare_claim(&[valid, invalid])
        .is_err());
    for pool in harness.root.dynamic_pools.pools.values() {
        let state = pool.state.lock().unwrap();
        assert_eq!(state.allocator.free_bytes, 64);
        assert_eq!(state.chunks.values().next().unwrap().live_segments, 0);
    }

    let valid_requests = harness
        .pool_ids
        .iter()
        .map(|pool_id| {
            let pool = &harness.root.dynamic_pools.pools[pool_id];
            evaluated_request(pool, 64)
        })
        .collect::<Vec<_>>();
    let BackingPrepareDecision::Prepared(prepared) = harness
        .root
        .dynamic_pools
        .prepare_claim(&valid_requests)
        .unwrap()
    else {
        panic!("both initialized pools must prepare")
    };
    drop(prepared);
    for pool in harness.root.dynamic_pools.pools.values() {
        let state = pool.state.lock().unwrap();
        assert_eq!(state.allocator.free_bytes, 64);
        assert_eq!(state.chunks.values().next().unwrap().live_segments, 0);
    }
}

#[test]
fn nth_pool_allocator_fault_rolls_back_prior_pool_and_poison_closes_faulted_pool() {
    let combined = combine_catalogs(&[
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'a',
            1,
            128,
            TestDemand::Fixed,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'b',
            1,
            128,
            TestDemand::Fixed,
        ),
    ]);
    let runtime = new_runtime(&combined, 256);
    let harness = harness(runtime, combined, 256, false);
    harness
        .root
        .maintenance_controller
        .initialize_pools(&harness.pool_ids)
        .unwrap();
    let faulted = &harness.root.dynamic_pools.pools[&harness.pool_ids[1]];
    faulted.state.lock().unwrap().allocator.by_offset.clear();
    let requests = harness
        .pool_ids
        .iter()
        .map(|pool_id| {
            let pool = &harness.root.dynamic_pools.pools[pool_id];
            evaluated_request(pool, 64)
        })
        .collect::<Vec<_>>();
    assert!(harness.root.dynamic_pools.prepare_claim(&requests).is_err());
    let first = harness.root.dynamic_pools.pools[&harness.pool_ids[0]]
        .state
        .lock()
        .unwrap();
    assert!(!first.poisoned);
    assert_eq!(first.allocator.free_bytes, 64);
    assert_eq!(first.allocator.by_offset.len(), 1);
    assert_eq!(first.chunks.values().next().unwrap().live_segments, 0);
    drop(first);
    let second = faulted.state.lock().unwrap();
    assert!(second.poisoned);
    assert_eq!(second.chunks.values().next().unwrap().live_segments, 0);
}

#[test]
fn contiguous_index_selects_from_ten_thousand_fragmented_holes_in_one_probe() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        64,
        TestDemand::Fixed,
    );
    let mut index = FreeExtentIndex::default();
    for hole in 0..10_000_u64 {
        index.insert_extent(1, 1, hole * 128, 64).unwrap();
    }
    assert_eq!(index.by_offset.len(), 10_000);
    let before = index.search_probes;
    let segment = index
        .allocate_contiguous(&catalog.pool_id, 64)
        .unwrap()
        .expect("one indexed hole fits");
    assert_eq!(index.search_probes - before, 1);
    assert_eq!(segment.length_bytes(), 64);
    assert_eq!(index.by_offset.len(), 9_999);
}

#[test]
fn paged_internal_fault_restores_segments_selected_earlier_in_the_same_call() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        128,
        TestDemand::Fixed,
    );
    let mut index = FreeExtentIndex::default();
    index.insert_extent(1, 1, 0, 64).unwrap();
    index.insert_extent(2, 2, 0, 64).unwrap();
    index.by_offset.get_mut(&(2, 0)).unwrap().length_bytes = 32;
    assert!(index.allocate_paged(&catalog.pool_id, 128, 64).is_err());
    assert_eq!(index.free_bytes, 128);
    assert_eq!(index.by_offset.len(), 2);
    assert_eq!(index.by_offset[&(1, 0)].length_bytes, 64);
    assert!(index.by_size.contains(&(64, 1, 1, 0)));
}
