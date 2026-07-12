use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, Weak};
use std::time::Duration;

const EXPECTED_CASES: usize = 311;
const DEVICE_GLOBAL_CAPACITY_CASES: usize = 20;
const RESOURCE_CAPACITY_TEST_MAXIMUM_CLAIMS: usize = 16;
const RESOURCE_CAPACITY_CONCURRENT_WORKERS: usize = 1;
const MAX_RESOURCE_TEST_CONCURRENT_WORKERS: usize = 4;

const _: () = assert!(
    RESOURCE_CAPACITY_CONCURRENT_WORKERS == 1
        && RESOURCE_CAPACITY_CONCURRENT_WORKERS <= MAX_RESOURCE_TEST_CONCURRENT_WORKERS
);

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn one_token_span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[1], 0..1).unwrap()
}

fn one_token_work() -> ResourceWorkShape {
    ResourceWorkShape::single(one_token_span()).unwrap()
}

fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

fn contiguous_storage_bindings(
    operation: &OperationDescriptor,
) -> Vec<ProviderStorageBindingRequirement> {
    operation
        .inputs
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal as u32,
                DynamicStorageRequirement::contiguous(),
            )
        })
        .chain(operation.outputs.iter().enumerate().map(|(ordinal, _)| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                ordinal as u32,
                DynamicStorageRequirement::contiguous(),
            )
        }))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TestConfig {
    width: u64,
}

#[derive(Default)]
struct TestFamily;

impl ModelFamilyProvider for TestFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.resource-contract"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("external-metadata.resource-contract")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("external-metadata.resource-contract"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: TestConfig = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        if config.width != 4 {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "width".to_owned(),
                reason: "resource fixture requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.resource-dense"),
            layout_id: id("weight-layout.resource-dense"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component"),
                role: WeightComponentRole::Values,
                external_names: vec!["weight.bin".to_owned()],
                dimensions: vec![config.width],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![config.width],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component"),
                },
                required: true,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.main"),
                    operation_id: id("operation.main"),
                    required_version: ContractVersion::new(1, 0),
                    inputs: vec![id("value.input"), id("value.weight"), id("value.state")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![StateSpec {
                id: id("state.cache"),
                value_id: id("value.state"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
            }],
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![id("value.output")],
        )
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".to_owned(),
                source_file: "template.json".to_owned(),
                sha256: sha('a'),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

fn tensor_contract(
    element_type: ElementType,
    access: TensorAccess,
    alias: AliasPolicy,
) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(4)],
        BTreeSet::from([element_type]),
        vec![LayoutConstraint::Contiguous],
        access,
        alias,
    )
    .unwrap()
}

fn operation() -> OperationDescriptor {
    OperationDescriptor {
        id: id("operation.main"),
        version: ContractVersion::new(1, 0),
        inputs: vec![
            tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
            tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
            tensor_contract(ElementType::U8, TensorAccess::Read, AliasPolicy::NoAlias),
        ],
        outputs: vec![tensor_contract(
            ElementType::F32,
            TensorAccess::Write,
            AliasPolicy::NoAlias,
        )],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            persistent: ResourcePresenceRequirement::Required,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([id("capability.compute")]),
        },
        profile_phase: ProfilePhase::Decode,
    }
}

fn catalog() -> CapabilityCatalog {
    catalog_with_runtime_fingerprint(sha('d'))
}

fn catalog_with_runtime_fingerprint(
    runtime_implementation_fingerprint: String,
) -> CapabilityCatalog {
    let operation = operation();
    let device_id: DeviceId = id("device.reference.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.reference"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        ContractVersion::new(1, 0),
        device_id.clone(),
        capabilities.clone(),
        BTreeSet::from([id("weight-format.resource-dense")]),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation),
        "resource-estimator.reference",
        ContractVersion::new(1, 0),
        sha('b'),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.reference"),
        ContractVersion::new(1, 0),
        sha('e'),
        device_id.clone(),
        capabilities.clone(),
    )
    .unwrap();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id,
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint,
            capabilities,
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        vec![operation.clone()],
        BTreeMap::from([(operation.id.clone(), vec![provider])]),
        vec![engine],
    )
    .unwrap()
}

struct TestOperationContract {
    descriptor: OperationDescriptor,
}

impl OperationContract for TestOperationContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.descriptor.inputs || outputs != self.descriptor.outputs {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "resource fixture operation signature mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

struct TestEstimator {
    descriptor: OperationProviderDescriptor,
}

impl OperationResourceEstimator for TestEstimator {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            Some(ProviderWorkspaceRequirement::new(
                64,
                16,
                ProviderWorkspaceScope::Invocation,
                DynamicStorageRequirement::contiguous(),
            )?),
            Some(ProviderWorkspaceRequirement::new(
                32,
                16,
                ProviderWorkspaceScope::Plan,
                DynamicStorageRequirement::contiguous(),
            )?),
        ))
    }
}

impl OperationProvider<TestRuntime> for TestEstimator {
    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

fn operation_registry(catalog: &CapabilityCatalog) -> OperationRuntimeRegistry<TestRuntime> {
    OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: catalog.operation(&id("operation.main")).unwrap().clone(),
        })],
        vec![Box::new(TestEstimator {
            descriptor: catalog.providers_for(&id("operation.main")).unwrap()[0].clone(),
        })],
    )
    .unwrap()
}

fn policy() -> ResolvedRuntimePolicy {
    policy_with_memory(4096, 128, 3)
}

fn policy_with_memory(
    capacity_bytes: u64,
    reserve_bytes: u64,
    maximum_active_sequences: u32,
) -> ResolvedRuntimePolicy {
    policy_with_memory_id(
        "runtime-policy.resource-test",
        capacity_bytes,
        reserve_bytes,
        maximum_active_sequences,
    )
}

fn policy_with_memory_id(
    policy_id: &str,
    capacity_bytes: u64,
    reserve_bytes: u64,
    maximum_active_sequences: u32,
) -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        policy_id,
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes,
            reserve_bytes,
            maximum_active_sequences,
            dynamic_storage_profile_order: vec![contiguous_storage_profile()],
        },
        AdmissionPolicy {
            maximum_queue_depth: 8,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
    )
    .unwrap()
}

fn resolved_tensor(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn binding(
    value_id: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
    access: TensorAccess,
    usage: BufferUsage,
    resource_id: &str,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id(value_id),
        role,
        ordinal,
        resolved_tensor(element_type),
        access,
        AliasPolicy::NoAlias,
        usage,
        ResolvedValueStorage::single(
            id(resource_id),
            0,
            4 * element_type.size_bytes(),
            element_type,
        )
        .unwrap(),
    )
    .unwrap()
}

fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &ResolvedRuntimePolicy,
    planning: &OperationPlanningHandle<'_>,
) -> PlanNodeResolution {
    let weight_storage = ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
        Some(id("weight.component")),
        id("resource.weight"),
        0,
        16,
        ElementType::F32,
    )
    .unwrap()])
    .unwrap();
    let values = vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            ElementType::F32,
            TensorAccess::Read,
            BufferUsage::Activations,
            "resource.input",
        ),
        ResolvedValueBinding::new(
            id("value.weight"),
            ResolvedValueRole::Input,
            1,
            resolved_tensor(ElementType::F32),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Weights,
            weight_storage,
        )
        .unwrap(),
        binding(
            "value.state",
            ResolvedValueRole::Input,
            2,
            ElementType::U8,
            TensorAccess::Read,
            BufferUsage::State,
            "resource.state",
        ),
        binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            ElementType::F32,
            TensorAccess::Write,
            BufferUsage::Activations,
            "resource.output",
        ),
    ];
    PlanNodeResolution::resolve(
        family,
        catalog,
        policy,
        planning,
        id("node.main"),
        values,
        BTreeSet::new(),
        None,
    )
    .unwrap()
}

fn execution_plan() -> ExecutionPlan {
    execution_plan_with_policy(policy())
}

fn execution_plan_with_policy(policy: ResolvedRuntimePolicy) -> ExecutionPlan {
    execution_plan_with_policy_and_runtime_fingerprint(policy, sha('d'))
}

fn execution_plan_with_policy_and_runtime_fingerprint(
    policy: ResolvedRuntimePolicy,
    runtime_implementation_fingerprint: String,
) -> ExecutionPlan {
    try_execution_plan_with_policy_and_runtime_fingerprint(
        policy,
        runtime_implementation_fingerprint,
    )
    .unwrap()
}

fn try_execution_plan_with_policy_and_runtime_fingerprint(
    policy: ResolvedRuntimePolicy,
    runtime_implementation_fingerprint: String,
) -> Result<ExecutionPlan, VNextError> {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog_with_runtime_fingerprint(runtime_implementation_fingerprint);
    let registry = operation_registry(&catalog);
    let planning = registry.planning();
    let resolution = node_resolution(&family, &catalog, &policy, &planning);
    ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
}

#[derive(Debug)]
struct TestBuffer {
    descriptor: BufferDescriptor,
    marker: String,
    drop_trace: Weak<Mutex<Trace>>,
    backend_lifetime: Weak<()>,
}

impl Drop for TestBuffer {
    fn drop(&mut self) {
        if let Some(trace) = self.drop_trace.upgrade() {
            let mut trace = trace.lock().unwrap();
            trace.buffer_drops += 1;
            if self.backend_lifetime.upgrade().is_none() {
                trace.buffer_drops_after_backend += 1;
            }
        }
    }
}

#[derive(Debug)]
struct TestStream {
    synchronize_count: u64,
    state: StreamState,
    drop_trace: Weak<Mutex<Trace>>,
}

impl Default for TestStream {
    fn default() -> Self {
        Self {
            synchronize_count: 0,
            state: StreamState::Ready,
            drop_trace: Weak::new(),
        }
    }
}

impl Drop for TestStream {
    fn drop(&mut self) {
        if let Some(trace) = self.drop_trace.upgrade() {
            trace.lock().unwrap().stream_drops += 1;
        }
    }
}

#[derive(Debug, Clone)]
struct TestRuntimeError(&'static str);

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvalidCommit {
    Descriptor,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PostAllocationBehavior {
    ForgetThenError,
    DropThenError,
    Panic,
}

#[derive(Default)]
struct Trace {
    calls: Vec<String>,
    runtime_allocate_calls: u64,
    drift_on_allocate: bool,
    drift_on_create_stream: bool,
    drift_on_submit: bool,
    drift_on_synchronize: bool,
    synchronize_failures: u32,
    synchronize_returns_not_ready: bool,
    panic_on_stream_state: bool,
    synchronize_block: Option<(Arc<Barrier>, Arc<Barrier>)>,
    runtime_synchronize_calls: u64,
    failures: BTreeMap<String, u32>,
    invalid_commits: BTreeMap<String, InvalidCommit>,
    post_allocation: BTreeMap<String, PostAllocationBehavior>,
    abandon: Vec<ResourceAbandonSignal>,
    quarantine_sizes: Vec<usize>,
    quarantine_actual_mismatch: Vec<bool>,
    durable_ownership: Vec<ResourcePoolOwnership<TestRuntime>>,
    abandon_claimed_bytes: Vec<u64>,
    abandon_buffer_counts: Vec<usize>,
    retain_ownership: bool,
    panic_on_abandon: bool,
    buffer_drops: u64,
    buffer_drops_after_backend: u64,
    stream_drops: u64,
}

#[derive(Clone)]
struct TestRuntime {
    descriptor: DeviceDescriptor,
    alternate_descriptor: DeviceDescriptor,
    use_alternate_descriptor: Arc<AtomicBool>,
    trace: Arc<Mutex<Trace>>,
    backend_lifetime: Arc<()>,
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = ();
    type Fence = ();
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        if self.use_alternate_descriptor.load(Ordering::Acquire) {
            &self.alternate_descriptor
        } else {
            &self.descriptor
        }
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let drift = {
            let mut trace = self.trace.lock().unwrap();
            trace.runtime_allocate_calls += 1;
            trace.drift_on_allocate
        };
        if drift {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        let invalid = self
            .trace
            .lock()
            .unwrap()
            .invalid_commits
            .get(permit.resource_id().as_str())
            .copied();
        let request = permit.into_request();
        let mut descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        match invalid {
            Some(InvalidCommit::Descriptor) => descriptor.size_bytes += 1,
            Some(InvalidCommit::Generation) => {
                descriptor.resource_id = id("resource.runtime-wrong-generation")
            }
            None => {}
        }
        Ok(TestBuffer {
            marker: request.resource_id().to_string(),
            descriptor,
            drop_trace: Arc::downgrade(&self.trace),
            backend_lifetime: Arc::downgrade(&self.backend_lifetime),
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        if self.trace.lock().unwrap().drift_on_create_stream {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        Ok(TestStream {
            drop_trace: Arc::downgrade(&self.trace),
            ..TestStream::default()
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        let panic_on_stream_state = {
            let mut trace = self.trace.lock().unwrap();
            std::mem::take(&mut trace.panic_on_stream_state)
        };
        if panic_on_stream_state {
            panic!("injected stream state panic");
        }
        stream.state
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
        stream: &mut Self::Stream,
        _command: Self::Command,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        if self.trace.lock().unwrap().drift_on_submit {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        stream.state = StreamState::Submitted;
        Ok(())
    }

    fn query_fence(&self, _fence: &Self::Fence) -> FenceQuery<Self::Error> {
        FenceQuery::Terminal(DeviceTerminal::Succeeded)
    }

    fn wait_fence(
        &self,
        _fence: &Self::Fence,
    ) -> Result<DeviceTerminal<Self::Error>, FenceIndeterminate<Self::Error>> {
        Ok(DeviceTerminal::Succeeded)
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        let (drift, fail, returns_not_ready, block) = {
            let mut trace = self.trace.lock().unwrap();
            trace.runtime_synchronize_calls += 1;
            let fail = trace.synchronize_failures > 0;
            trace.synchronize_failures = trace.synchronize_failures.saturating_sub(1);
            (
                trace.drift_on_synchronize,
                fail,
                trace.synchronize_returns_not_ready,
                trace.synchronize_block.take(),
            )
        };
        if let Some((entered, release)) = block {
            entered.wait();
            release.wait();
        }
        stream.synchronize_count += 1;
        if drift {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        if fail {
            stream.state = StreamState::Failed;
            Err(TestRuntimeError("injected synchronize failure"))
        } else {
            stream.state = if returns_not_ready {
                StreamState::Submitted
            } else {
                StreamState::Ready
            };
            Ok(())
        }
    }

    fn readback(
        &self,
        _stream: &mut Self::Stream,
        _source: &Self::Buffer,
        _region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![0; output_layout.byte_len().unwrap() as usize])
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("test_runtime", error.to_string(), false)
    }
}

#[derive(Clone)]
struct TestDriver {
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    device_capacity_bytes: u64,
    trace: Arc<Mutex<Trace>>,
    runtime: Arc<TestRuntime>,
}

impl fmt::Debug for TestDriver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TestDriver")
            .field("device_id", &self.device_id)
            .field(
                "device_runtime_implementation_fingerprint",
                &self.device_runtime_implementation_fingerprint,
            )
            .field("device_capacity_bytes", &self.device_capacity_bytes)
            .finish_non_exhaustive()
    }
}

impl TestDriver {
    fn record(&self, action: &str, resource: Option<&ResourceId>) -> bool {
        let key = resource
            .map(|resource| format!("{action}:{}", resource.as_str()))
            .unwrap_or_else(|| action.to_owned());
        let mut trace = self.trace.lock().unwrap();
        trace.calls.push(key.clone());
        let remaining = trace.failures.entry(key).or_default();
        if *remaining == 0 {
            false
        } else {
            *remaining -= 1;
            true
        }
    }

    fn failure(action: &str) -> ResourceDriverFailure {
        ResourceDriverFailure::new(
            FailureEnvelope::new(
                FailureDomain::Resource,
                format!("{action}_failed"),
                format!("injected {action} failure"),
                true,
            )
            .unwrap(),
        )
        .unwrap()
    }
}

impl ResourceTransactionDriver for TestDriver {
    type Buffer = TestBuffer;
    type Runtime = TestRuntime;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    fn reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        assert!(matches!(
            context.allocate(&request),
            Err(DeviceAllocationError::Contract(_))
        ));
        if self.record("reserve", Some(reservation.resource_id())) {
            Err(Self::failure("reserve"))
        } else {
            Ok(())
        }
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        if self.record("commit", Some(reservation.resource_id())) {
            return Err(Self::failure("commit"));
        }
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        let receipt = context
            .allocate(&request)
            .map_err(|_| Self::failure("commit-allocation"))?;
        assert!(matches!(
            context.allocate(&request),
            Err(DeviceAllocationError::Contract(_))
        ));
        assert_eq!(receipt.resource_id(), reservation.resource_id());
        assert_eq!(receipt.generation(), reservation.generation());
        let behavior = self
            .trace
            .lock()
            .unwrap()
            .post_allocation
            .remove(reservation.resource_id().as_str());
        match behavior {
            Some(PostAllocationBehavior::ForgetThenError) => {
                std::mem::forget(receipt);
                Err(Self::failure("commit-after-allocation"))
            }
            Some(PostAllocationBehavior::DropThenError) => {
                drop(receipt);
                Err(Self::failure("commit-after-allocation"))
            }
            Some(PostAllocationBehavior::Panic) => {
                drop(receipt);
                panic!("injected panic after allocation")
            }
            None => Ok(receipt),
        }
    }

    fn compensate_reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        assert!(matches!(
            context.allocate(&request),
            Err(DeviceAllocationError::Contract(_))
        ));
        if self.record("undo-reserve", Some(reservation.resource_id())) {
            Err(Self::failure("undo-reserve"))
        } else {
            Ok(())
        }
    }

    fn compensate_commit_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.marker, reservation.resource_id().as_str());
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        assert!(matches!(
            context.allocate(&request),
            Err(DeviceAllocationError::Contract(_))
        ));
        if self.record("undo-commit", Some(reservation.resource_id())) {
            Err(Self::failure("undo-commit"))
        } else {
            Ok(())
        }
    }

    fn rollback_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        if self.record("rollback", Some(reservation.resource_id())) {
            Err(Self::failure("rollback"))
        } else {
            Ok(())
        }
    }

    fn release_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.marker, reservation.resource_id().as_str());
        if self.record("release", Some(reservation.resource_id())) {
            Err(Self::failure("release"))
        } else {
            Ok(())
        }
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        expected: &ResourceReservation,
        actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(actual.buffer().marker, expected.resource_id().as_str());
        if self.record("reconcile", Some(expected.resource_id())) {
            Err(Self::failure("reconcile"))
        } else {
            Ok(())
        }
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        let mismatch = ownership.buffers().iter().any(|view| {
            view.resource_id() != view.actual_resource_id()
                || view.generation() != view.actual_generation()
                || view.expected_descriptor() != view.actual_descriptor()
        });
        let failed = self.record("quarantine", None);
        let mut trace = self.trace.lock().unwrap();
        trace.quarantine_sizes.push(ownership.buffers().len());
        trace.quarantine_actual_mismatch.push(mismatch);
        drop(trace);
        if failed {
            Err(ResourceOwnershipTransferFailure::new(
                Self::failure("quarantine"),
                ownership,
            ))
        } else {
            let mut trace = self.trace.lock().unwrap();
            if trace.retain_ownership {
                trace.durable_ownership.push(ownership);
            } else {
                drop(trace);
                drop(ownership);
            }
            Ok(())
        }
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        let mut trace = self.trace.lock().unwrap();
        trace.abandon.push(
            ownership
                .abandon_signal()
                .expect("abandon ownership has a signal")
                .clone(),
        );
        trace.abandon_claimed_bytes.push(ownership.claimed_bytes());
        trace.abandon_buffer_counts.push(ownership.buffers().len());
        let panic_on_abandon = trace.panic_on_abandon;
        if panic_on_abandon {
            drop(trace);
            panic!("injected abandon callback panic");
        }
        if trace.retain_ownership {
            trace.durable_ownership.push(ownership);
        } else {
            drop(trace);
            drop(ownership);
        }
    }
}

fn configured_driver(
    plan: &ExecutionPlan,
    failures: &[(&str, u32)],
    invalid_commits: &[(ResourceId, InvalidCommit)],
) -> (TestDriver, Arc<Mutex<Trace>>) {
    let trace = Arc::new(Mutex::new(Trace {
        failures: failures
            .iter()
            .map(|(key, count)| ((*key).to_owned(), *count))
            .collect(),
        invalid_commits: invalid_commits
            .iter()
            .map(|(resource, mode)| (resource.to_string(), *mode))
            .collect(),
        ..Trace::default()
    }));
    let descriptor = DeviceDescriptor {
        id: plan.payload().device_id().clone(),
        class: DeviceClass::Reference,
        ordinal: 0,
        total_memory_bytes: plan.payload().memory().device_capacity_bytes(),
        runtime_implementation_fingerprint: plan
            .payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
    };
    let mut alternate_descriptor = descriptor.clone();
    alternate_descriptor.runtime_implementation_fingerprint =
        if descriptor.runtime_implementation_fingerprint == sha('d') {
            sha('f')
        } else {
            sha('d')
        };
    let runtime = Arc::new(TestRuntime {
        descriptor,
        alternate_descriptor,
        use_alternate_descriptor: Arc::new(AtomicBool::new(false)),
        trace: trace.clone(),
        backend_lifetime: Arc::new(()),
    });
    (
        TestDriver {
            device_id: plan.payload().device_id().clone(),
            device_runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            device_capacity_bytes: plan.payload().memory().device_capacity_bytes(),
            trace: trace.clone(),
            runtime,
        },
        trace,
    )
}

fn sequence_runtime(plan: &ExecutionPlan) -> Arc<TestRuntime> {
    let (_, trace) = configured_driver(plan, &[], &[]);
    let descriptor = DeviceDescriptor {
        id: plan.payload().device_id().clone(),
        class: DeviceClass::Reference,
        ordinal: 0,
        total_memory_bytes: plan.payload().memory().device_capacity_bytes(),
        runtime_implementation_fingerprint: plan
            .payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
    };
    let mut alternate_descriptor = descriptor.clone();
    alternate_descriptor.runtime_implementation_fingerprint =
        if descriptor.runtime_implementation_fingerprint == sha('d') {
            sha('f')
        } else {
            sha('d')
        };
    Arc::new(TestRuntime {
        descriptor,
        alternate_descriptor,
        use_alternate_descriptor: Arc::new(AtomicBool::new(false)),
        trace,
        backend_lifetime: Arc::new(()),
    })
}

fn transaction(
    plan: &ExecutionPlan,
    driver: TestDriver,
    suffix: &str,
) -> ResourceTransaction<TestDriver, TransactionNew> {
    let provisioning_request: RequestIdentity = id(format!("request.provision.{suffix}"));
    let permit = required_static(plan, Arc::clone(driver.runtime()), provisioning_request).unwrap();
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id(format!("run.resource.{suffix}")),
        id(format!("transaction.resource.{suffix}")),
    );
    ResourceTransaction::begin(driver, identity, permit).unwrap()
}

fn plan_runtime(
    plan: &ExecutionPlan,
    driver: TestDriver,
    suffix: &str,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    let committed = transaction(plan, driver, suffix)
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    match committed.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    }
}

fn admit_logical_request(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedRequestResources<TestRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = root.trusted_runtime_binding().unwrap();
    for attempt in 0..=3 {
        match binding
            .try_admit_request(request.clone(), id(run_id), id(request_id))
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(resources) => return resources,
            RequestResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                root.maintain_for_deferred(&deferred).unwrap();
            }
            RequestResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("request backing did not converge after bounded maintenance")
            }
            RequestResourceAdmissionDecision::Deferred(_) => {
                panic!("logical request unexpectedly deferred")
            }
            RequestResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("logical request unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded request admission loop always returns or panics")
}

fn admit_logical_sequence(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let request = admit_logical_request(root, run_id, request_id);
    let sequence = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match request.try_admit_sequence(sequence.clone()).unwrap() {
            SequenceResourceAdmissionDecision::Admitted(resources) => return resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                root.maintain_for_deferred(&deferred).unwrap();
            }
            SequenceResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("sequence backing did not converge after bounded maintenance")
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("logical sequence unexpectedly deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("logical sequence unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded sequence admission loop always returns or panics")
}

fn close_plan_runtime(root: Arc<PlanRuntimeResources<TestRuntime>>) -> PlanRuntimeCloseReceipt {
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
}

fn admit_resources(
    plan: &ExecutionPlan,
    request_id: RequestIdentity,
) -> Result<StaticProvisioningPermit<TestRuntime>, VNextError> {
    required_static(plan, sequence_runtime(plan), request_id)
}

fn required_static(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
    request_id: RequestIdentity,
) -> Result<StaticProvisioningPermit<TestRuntime>, VNextError> {
    let ProvisionedPlanParts { provisioning } =
        plan.provision_static(runtime, request_id)?.into_parts();
    match provisioning {
        StaticProvisioning::Required(permit) => Ok(permit),
        StaticProvisioning::NoStatic(_) => Err(VNextError::InvalidExecutionPlan {
            reason: "test plan unexpectedly has no physical provisioning".to_owned(),
        }),
    }
}

fn plan_resources(plan: &ExecutionPlan) -> Vec<ResourceId> {
    plan.payload()
        .memory()
        .static_allocations()
        .iter()
        .map(|allocation| allocation.resource_id().clone())
        .collect()
}

fn failure_key(action: &str, resource: &ResourceId) -> String {
    format!("{action}:{}", resource.as_str())
}

fn calls(trace: &Arc<Mutex<Trace>>, prefix: &str) -> Vec<String> {
    trace
        .lock()
        .unwrap()
        .calls
        .iter()
        .filter(|call| call.starts_with(prefix))
        .cloned()
        .collect()
}

fn expect_err<T, E>(result: Result<T, E>) -> E {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
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

fn rehash_plan_json(value: &mut Value) {
    let payload = value["payload"].as_object_mut().unwrap();
    payload.remove("plan_id");
    let material = canonical_json(Value::Object(payload.clone()));
    let bytes = serde_json::to_vec(&material).unwrap();
    let digest = format!("{:x}", Sha256::digest(bytes));
    value["payload"]["plan_id"] = json!(format!("plan/sha256/{digest}"));
    value["plan_hash"] = json!(digest);
}

fn runtime_implementation_authority_contract(plan: &ExecutionPlan, passed: &mut usize) {
    check(
        passed,
        plan.payload().device_runtime_implementation_fingerprint() == sha('d'),
    );
    let propagation = admit_resources(plan, id("request.runtime-fingerprint.propagation")).unwrap();
    check(
        passed,
        propagation
            .binding()
            .device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );
    check(
        passed,
        propagation
            .binding()
            .pool_identity()
            .device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );
    drop(propagation);

    let mut wrong_runtime = sequence_runtime(plan);
    let wrong_trace = Arc::clone(&wrong_runtime.trace);
    Arc::get_mut(&mut wrong_runtime)
        .unwrap()
        .descriptor
        .runtime_implementation_fingerprint = sha('f');
    check(
        passed,
        required_static(
            plan,
            wrong_runtime,
            id("request.runtime-fingerprint.wrong-admission"),
        )
        .is_err(),
    );
    check(
        passed,
        wrong_trace.lock().unwrap().runtime_allocate_calls == 0,
    );

    let admitted_runtime = sequence_runtime(plan);
    let exact = required_static(
        plan,
        Arc::clone(&admitted_runtime),
        id("request.runtime-instance.exact-admission"),
    )
    .unwrap();
    let exact_identity = ResourceTransactionIdentity::for_admission(
        exact.binding(),
        id("run.runtime-instance.impostor-driver"),
        id("transaction.runtime-instance.impostor-driver"),
    );
    let (impostor_driver, impostor_trace) = configured_driver(plan, &[], &[]);
    check(
        passed,
        ResourceTransaction::begin(impostor_driver, exact_identity, exact).is_err(),
    );
    check(passed, impostor_trace.lock().unwrap().calls.is_empty());

    let (drifting_driver, drifting_trace) = configured_driver(plan, &[], &[]);
    drifting_trace.lock().unwrap().drift_on_allocate = true;
    let reserved = transaction(plan, drifting_driver, "descriptor-drift-allocation")
        .reserve()
        .unwrap();
    let rejected_commit = reserved.commit();
    check(passed, rejected_commit.is_err());
    drop(rejected_commit);
    check(
        passed,
        drifting_trace.lock().unwrap().runtime_allocate_calls == 1,
    );

    let alternate_plan = execution_plan_with_policy_and_runtime_fingerprint(policy(), sha('f'));
    check(
        passed,
        alternate_plan
            .payload()
            .device_runtime_implementation_fingerprint()
            == sha('f'),
    );
    let anchor = admit_resources(plan, id("request.runtime-fingerprint.account-anchor")).unwrap();
    check(
        passed,
        admit_resources(
            &alternate_plan,
            id("request.runtime-fingerprint.account-conflict"),
        )
        .is_err(),
    );
    drop(anchor);
    check(
        passed,
        admit_resources(
            &alternate_plan,
            id("request.runtime-fingerprint.account-after-drop"),
        )
        .is_ok(),
    );

    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let policy = policy();
    let registry = operation_registry(&catalog);
    let planning = registry.planning();
    let resolution = node_resolution(&family, &catalog, &policy, &planning);
    let mut wire = serde_json::to_value(plan).unwrap();
    wire["payload"]["device_runtime_implementation_fingerprint"] = json!(sha('f'));
    rehash_plan_json(&mut wire);
    check(
        passed,
        ExecutionPlan::from_json_validated(
            &serde_json::to_vec(&wire).unwrap(),
            &family,
            &catalog,
            &policy,
            vec![resolution],
        )
        .is_err(),
    );
}

fn device_global_capacity_contract(base_plan: &ExecutionPlan, passed: &mut usize) {
    let expected_peak = base_plan.payload().memory().static_bytes();
    let scaled = |factor: u64| {
        expected_peak
            .checked_mul(factor)
            .expect("bounded resource test capacity does not overflow")
    };
    let plan = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.capacity-primary",
        scaled(20),
        scaled(4),
        1_000,
    ));
    let second_plan = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.second-plan",
        scaled(20),
        scaled(3),
        1_000,
    ));
    let peak = plan.payload().memory().static_bytes();
    assert_eq!(peak, expected_peak);
    let raw_capacity = plan.payload().memory().device_capacity_bytes();
    let effective_usable_capacity = plan
        .payload()
        .memory()
        .usable_capacity_bytes()
        .min(second_plan.payload().memory().usable_capacity_bytes());
    let maximum_claims = (effective_usable_capacity / peak) as usize;
    assert_eq!(maximum_claims, RESOURCE_CAPACITY_TEST_MAXIMUM_CLAIMS);
    check(passed, maximum_claims >= 2);
    check(passed, effective_usable_capacity < raw_capacity);
    check(
        passed,
        second_plan.payload().device_id() == plan.payload().device_id()
            && second_plan.payload().memory().usable_capacity_bytes()
                != plan.payload().memory().usable_capacity_bytes(),
    );
    check(
        passed,
        try_execution_plan_with_policy_and_runtime_fingerprint(
            policy_with_memory_id(
                "runtime-policy.resource-test.own-usable-reject",
                scaled(2),
                scaled(2) - 1,
                1_000,
            ),
            sha('d'),
        )
        .is_err(),
    );

    let mut permits = Vec::new();
    for index in 0..maximum_claims {
        let source = if index % 2 == 0 { &plan } else { &second_plan };
        permits.push(
            admit_resources(source, id(format!("request.capacity.multi-plan.{index}"))).unwrap(),
        );
    }
    check(
        passed,
        admit_resources(&second_plan, id("request.capacity.over-limit")).is_err(),
    );
    check(
        passed,
        permits
            .windows(2)
            .all(|pair| pair[0].binding().pool_id() != pair[1].binding().pool_id()),
    );
    drop(permits.pop());
    let replacement =
        admit_resources(&second_plan, id("request.capacity.after-permit-drop")).unwrap();
    check(passed, replacement.binding().admitted_bytes() == peak);
    drop(replacement);
    drop(permits);

    let held = admit_resources(&plan, id("request.capacity.metadata-anchor")).unwrap();
    let different_capacity = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.different-capacity",
        scaled(18),
        scaled(3),
        1_000,
    ));
    let different_held = admit_resources(
        &different_capacity,
        id("request.capacity.different-usable-a-then-b"),
    )
    .unwrap();
    check(passed, different_held.binding().admitted_bytes() == peak);
    drop(different_held);
    drop(held);
    let reverse_anchor = admit_resources(
        &different_capacity,
        id("request.capacity.different-usable-b-anchor"),
    )
    .unwrap();
    check(
        passed,
        admit_resources(&plan, id("request.capacity.different-usable-b-then-a")).is_ok(),
    );
    drop(reverse_anchor);

    let permit = admit_resources(&plan, id("request.capacity.begin-failure")).unwrap();
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.capacity.begin-failure"),
        id("transaction.capacity.begin-failure"),
    );
    let (mut wrong_driver, _) = configured_driver(&plan, &[], &[]);
    wrong_driver.device_capacity_bytes -= 1;
    check(
        passed,
        ResourceTransaction::begin(wrong_driver, identity, permit).is_err(),
    );
    let mut after_begin_failure = Vec::new();
    for index in 0..maximum_claims {
        after_begin_failure.push(
            admit_resources(&plan, id(format!("request.capacity.begin-return.{index}"))).unwrap(),
        );
    }
    check(passed, after_begin_failure.len() == maximum_claims);
    drop(after_begin_failure);

    let (driver, _) = configured_driver(&plan, &[], &[]);
    let rolled_back = transaction(&plan, driver, "capacity-terminal-rollback")
        .reserve()
        .unwrap()
        .rollback()
        .unwrap();
    let mut while_terminal_held = Vec::new();
    for index in 0..maximum_claims {
        while_terminal_held.push(
            admit_resources(
                &plan,
                id(format!("request.capacity.terminal-return.{index}")),
            )
            .unwrap(),
        );
    }
    check(passed, while_terminal_held.len() == maximum_claims);
    drop(while_terminal_held);
    drop(rolled_back);

    let first_release = failure_key("release", &plan_resources(&plan)[0]);
    let (driver, trace) = configured_driver(&plan, &[(first_release.as_str(), 1)], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "capacity-durable-quarantine")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let quarantined = expect_err(committed.release()).quarantine().unwrap();
    check(passed, trace.lock().unwrap().durable_ownership.len() == 1);
    let quarantine_owned_shape = {
        let trace = trace.lock().unwrap();
        trace.durable_ownership[0].claimed_bytes() == peak
            && trace.durable_ownership[0].buffers().len()
                == plan.payload().memory().static_allocations().len()
    };
    check(passed, quarantine_owned_shape);
    let mut under_quarantine = Vec::new();
    for index in 0..(maximum_claims - 1) {
        under_quarantine.push(
            admit_resources(&plan, id(format!("request.capacity.quarantined.{index}"))).unwrap(),
        );
    }
    check(
        passed,
        admit_resources(&plan, id("request.capacity.quarantine-still-claimed")).is_err(),
    );
    drop(under_quarantine);
    let cleaned = std::mem::take(&mut trace.lock().unwrap().durable_ownership);
    drop(cleaned);
    check(
        passed,
        admit_resources(&plan, id("request.capacity.quarantine-cleaned")).is_ok(),
    );
    drop(quarantined);

    let (driver, trace) = configured_driver(&plan, &[], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "capacity-durable-abandon")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    drop(committed);
    let abandon_owned_shape = {
        let trace = trace.lock().unwrap();
        trace.durable_ownership.len() == 1 && trace.abandon_claimed_bytes == [peak]
    };
    check(passed, abandon_owned_shape);
    check(
        passed,
        trace.lock().unwrap().durable_ownership[0].reason() == ResourceOwnershipReason::Abandon,
    );
    let cleaned = std::mem::take(&mut trace.lock().unwrap().durable_ownership);
    drop(cleaned);

    // Capacity cardinality is not an OS-thread cardinality. Fill every slot but
    // one serially, then use one bounded worker plus the calling test thread to
    // prove that exactly one contender can atomically take the final slot.
    let mut concurrent_prefill = Vec::with_capacity(maximum_claims - 1);
    for index in 0..(maximum_claims - 1) {
        let source = if index % 2 == 0 { &plan } else { &second_plan };
        concurrent_prefill.push(
            admit_resources(
                source,
                id(format!("request.capacity.concurrent-prefill.{index}")),
            )
            .unwrap(),
        );
    }
    let barrier = Arc::new(Barrier::new(RESOURCE_CAPACITY_CONCURRENT_WORKERS + 1));
    let results = std::thread::scope(|scope| {
        let worker_barrier = Arc::clone(&barrier);
        let worker = std::thread::Builder::new()
            .name("vnext-resource-capacity-contender".to_owned())
            .spawn_scoped(scope, move || {
                worker_barrier.wait();
                admit_resources(&second_plan, id("request.capacity.concurrent.worker"))
            })
            .expect("bounded resource-capacity worker starts");
        barrier.wait();
        let caller = admit_resources(&plan, id("request.capacity.concurrent.caller"));
        vec![
            caller,
            worker
                .join()
                .expect("bounded resource-capacity worker does not panic"),
        ]
    });
    let successful = results.iter().filter(|result| result.is_ok()).count();
    check(passed, successful == 1);
    check(passed, results.len() - successful == 1);
    drop(results);
    drop(concurrent_prefill);
}

#[test]
fn resource_capacity_concurrency_is_bounded() {
    let plan = execution_plan();
    let mut passed = 0;
    device_global_capacity_contract(&plan, &mut passed);
    assert_eq!(passed, DEVICE_GLOBAL_CAPACITY_CASES);
    println!("VNEXT RESOURCE CAPACITY THREAD BOUND PASS: {passed}/{DEVICE_GLOBAL_CAPACITY_CASES}");
}

fn admission_and_success(plan: &ExecutionPlan, passed: &mut usize) {
    let request_a: RequestIdentity = id("request.provision.admission-a");
    let request_b: RequestIdentity = id("request.provision.admission-b");
    let permit_a = admit_resources(plan, request_a.clone()).unwrap();
    let generation_a = permit_a.binding().admission_generation();
    check(
        passed,
        permit_a.binding().plan_id() == plan.payload().plan_id(),
    );
    check(passed, permit_a.binding().plan_hash() == plan.plan_hash());
    check(
        passed,
        permit_a.binding().pool_identity().plan_id() == plan.payload().plan_id()
            && permit_a.binding().pool_identity().plan_hash() == plan.plan_hash()
            && permit_a.binding().pool_identity().device_id() == plan.payload().device_id(),
    );
    check(
        passed,
        permit_a.binding().pool_identity().admission_generation() == generation_a
            && permit_a.binding().pool_id().get() == generation_a,
    );
    check(
        passed,
        permit_a.binding().admitted_bytes() == plan.payload().memory().static_bytes(),
    );
    check(
        passed,
        permit_a.binding().usable_capacity_bytes()
            == plan.payload().memory().usable_capacity_bytes(),
    );
    check(
        passed,
        permit_a.binding().maximum_active_sequences()
            == plan.payload().memory().maximum_active_sequences(),
    );
    check(
        passed,
        permit_a.reservations().reservations().len()
            == plan.payload().memory().static_allocations().len(),
    );
    let permit_b = admit_resources(plan, request_b).unwrap();
    check(
        passed,
        permit_b.binding().admission_generation() > generation_a,
    );

    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut wrong_device = driver.clone();
    wrong_device.device_id = id("device.reference.other");
    let wrong_device_identity = ResourceTransactionIdentity::for_admission(
        permit_a.binding(),
        id("run.wrong-device"),
        id("transaction.wrong-device"),
    );
    check(
        passed,
        ResourceTransaction::begin(wrong_device, wrong_device_identity, permit_a).is_err(),
    );

    let request_capacity: RequestIdentity = id("request.provision.wrong-capacity");
    let permit_capacity = admit_resources(plan, request_capacity.clone()).unwrap();
    let mut wrong_capacity = driver.clone();
    wrong_capacity.device_capacity_bytes -= 1;
    let wrong_capacity_identity = ResourceTransactionIdentity::for_admission(
        permit_capacity.binding(),
        id("run.wrong-capacity"),
        id("transaction.wrong-capacity"),
    );
    check(
        passed,
        ResourceTransaction::begin(wrong_capacity, wrong_capacity_identity, permit_capacity)
            .is_err(),
    );

    let request_identity: RequestIdentity = id("request.provision.identity-mismatch");
    let permit_identity = admit_resources(plan, request_identity).unwrap();
    let other_permit = admit_resources(plan, id("request.provision.other-pool")).unwrap();
    let mismatched_identity = ResourceTransactionIdentity::for_admission(
        other_permit.binding(),
        id("run.identity-mismatch"),
        id("transaction.identity-mismatch"),
    );
    check(
        passed,
        ResourceTransaction::begin(driver.clone(), mismatched_identity, permit_identity).is_err(),
    );
    drop(other_permit);

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "success").reserve().unwrap();
    let resource_count = plan.payload().memory().static_allocations().len();
    check(
        passed,
        reserved
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Reserved),
    );
    check(passed, reserved.receipts().len() == 1);
    check(
        passed,
        reserved.receipts()[0].records().len() == resource_count,
    );
    check(
        passed,
        reserved.latest_transition_validation_context().is_some(),
    );
    let committed = reserved.commit().unwrap();
    check(
        passed,
        committed.identity().pool_id() == committed.admission().pool_id(),
    );
    check(
        passed,
        committed.lease().entries().count() == resource_count,
    );
    check(
        passed,
        committed
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed),
    );
    let first = committed.lease().entries().next().unwrap();
    let first_id = first.resource_id().clone();
    let first_generation = first.generation();
    check(
        passed,
        plan.payload()
            .memory()
            .static_allocations()
            .iter()
            .any(|allocation| allocation.resource_id() == &first_id),
    );
    check(
        passed,
        first_generation == committed.admission().admission_generation(),
    );
    check(
        passed,
        first.size_bytes() > 0 && first.alignment_bytes().is_power_of_two(),
    );
    let released = committed.release().unwrap();
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(passed, trace.lock().unwrap().abandon.is_empty());
}

fn reverse_incremental_recovery(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    assert_eq!(resources.len(), 2);

    let reserve_failure = failure_key("reserve", &resources[1]);
    let undo_failure = failure_key("undo-reserve", &resources[0]);
    let failures = [(reserve_failure.as_str(), 1), (undo_failure.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let error = expect_err(transaction(plan, driver, "reserve-recovery").reserve());
    check(passed, error.failure().completed().len() == 1);
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state() == ResourceTransactionState::New,
    );
    let error = expect_err(error.recover());
    check(passed, error.failure().compensation().is_empty());
    check(
        passed,
        error.failure().recovery_failures().len() == 1
            && error.failure().recovery_failures()[0]
                .resource()
                .unwrap()
                .resource_id()
                == &resources[0],
    );
    let recovered = error.recover().unwrap();
    check(
        passed,
        recovered
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::New),
    );
    check(
        passed,
        calls(&trace, "undo-reserve")
            == [
                failure_key("undo-reserve", &resources[0]),
                failure_key("undo-reserve", &resources[0]),
            ],
    );
    let _terminal = recovered.reserve().unwrap().rollback().unwrap();

    let commit_failure = failure_key("commit", &resources[1]);
    let undo_commit_failure = failure_key("undo-commit", &resources[0]);
    let failures = [
        (commit_failure.as_str(), 1),
        (undo_commit_failure.as_str(), 1),
    ];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let reserved = transaction(plan, driver, "commit-recovery")
        .reserve()
        .unwrap();
    let error = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Recoverable(error) => error,
        ResourceCommitTransitionError::Poisoned(_) => panic!("driver error is recoverable"),
    };
    check(passed, error.failure().completed().len() == 1);
    check(
        passed,
        error.failure().ledger_after()[0].buffer_present()
            && !error.failure().ledger_after()[1].buffer_present(),
    );
    let error = expect_err(error.recover());
    check(passed, error.failure().compensation().is_empty());
    let reserved = error.recover().unwrap();
    check(
        passed,
        reserved
            .ledger_snapshot()
            .entries()
            .iter()
            .all(|entry| !entry.buffer_present()),
    );
    check(
        passed,
        calls(&trace, "undo-commit")
            == [
                failure_key("undo-commit", &resources[0]),
                failure_key("undo-commit", &resources[0]),
            ],
    );
    let _terminal = reserved.commit().unwrap().release().unwrap();
}

fn poison_reconcile_and_quarantine(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let malformed = resources[1].clone();

    let (driver, trace) =
        configured_driver(plan, &[], &[(malformed.clone(), InvalidCommit::Descriptor)]);
    let reserved = transaction(plan, driver, "poison-reconcile")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => panic!("bad descriptor must poison"),
    };
    check(
        passed,
        poisoned.failure().recovery_strategy() == ResourceRecoveryStrategy::ReconcileOrQuarantine,
    );
    check(
        passed,
        poisoned.failure().failure_point().unwrap().resource_id() == &malformed,
    );
    check(
        passed,
        poisoned.failure().failure_point().unwrap().actual_before()
            == ResourceTransactionState::Reserved,
    );
    check(passed, poisoned.failure().completed().len() == 1);
    check(
        passed,
        poisoned.failure().ledger_after()[0].transaction_state()
            == ResourceTransactionState::Committed,
    );
    check(
        passed,
        poisoned.failure().ledger_after()[1].transaction_state()
            == ResourceTransactionState::Reserved,
    );
    let recovery = poisoned.reconcile().unwrap();
    check(
        passed,
        recovery.failure().recovery_strategy() == ResourceRecoveryStrategy::ReverseCompensation,
    );
    let reserved = recovery.recover().unwrap();
    check(passed, calls(&trace, "reconcile").len() == 1);
    check(passed, calls(&trace, "undo-commit").len() == 1);
    let _terminal = reserved.rollback().unwrap();

    let (driver, trace) =
        configured_driver(plan, &[], &[(malformed.clone(), InvalidCommit::Generation)]);
    let reserved = transaction(plan, driver, "poison-quarantine")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => panic!("bad generation must poison"),
    };
    let quarantined = poisoned.quarantine().unwrap();
    check(
        passed,
        quarantined
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Quarantined),
    );
    let quarantine_receipt = quarantined.receipts().last().unwrap();
    check(
        passed,
        quarantine_receipt.records()[0].before() == ResourceTransactionState::Committed,
    );
    check(
        passed,
        quarantine_receipt.records()[1].before() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        quarantine_receipt
            .records()
            .iter()
            .all(|record| record.after() == ResourceTransactionState::Quarantined),
    );
    check(
        passed,
        quarantined
            .ledger_snapshot()
            .entries()
            .iter()
            .all(|entry| !entry.buffer_present()),
    );
    check(passed, trace.lock().unwrap().quarantine_sizes == [2]);
    check(
        passed,
        trace.lock().unwrap().quarantine_actual_mismatch == [true],
    );

    let reconcile_failure = failure_key("reconcile", &malformed);
    let failures = [(reconcile_failure.as_str(), 1)];
    let (driver, trace) =
        configured_driver(plan, &failures, &[(malformed, InvalidCommit::Descriptor)]);
    let reserved = transaction(plan, driver, "poison-reconcile-failure")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => unreachable!(),
    };
    let poisoned = expect_err(poisoned.reconcile());
    check(passed, poisoned.failure().recovery_failures().len() == 1);
    check(passed, poisoned.quarantine().is_ok());
    check(passed, calls(&trace, "quarantine").len() == 1);
}

fn forward_only_recovery(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    assert_eq!(resources.len(), 2);
    let rollback_first = failure_key("rollback", &resources[0]);
    let rollback_second = failure_key("rollback", &resources[1]);
    let failures = [(rollback_first.as_str(), 1), (rollback_second.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let reserved = transaction(plan, driver, "rollback-forward")
        .reserve()
        .unwrap();
    let error = expect_err(reserved.rollback());
    check(passed, error.failure().completed().is_empty());
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state() == ResourceTransactionState::Reserved,
    );
    let error = expect_err(error.complete());
    check(passed, error.failure().completed().len() == 1);
    let rolled_back = error.complete().unwrap();
    check(
        passed,
        rolled_back
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::RolledBack),
    );
    check(
        passed,
        calls(&trace, "rollback")
            == [
                failure_key("rollback", &resources[0]),
                failure_key("rollback", &resources[0]),
                failure_key("rollback", &resources[1]),
                failure_key("rollback", &resources[1]),
            ],
    );

    let release_first = failure_key("release", &resources[0]);
    let release_second = failure_key("release", &resources[1]);
    let failures = [(release_first.as_str(), 1), (release_second.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let committed = transaction(plan, driver, "release-forward")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let error = expect_err(committed.release());
    let failure_anchor = error.failure().clone();
    check(passed, failure_anchor.failure_id().get() != 0);
    check(passed, error.failure().completed().is_empty());
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state()
            == ResourceTransactionState::Committed,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state()
            == ResourceTransactionState::Committed,
    );
    let error = expect_err(error.complete());
    check(
        passed,
        error
            .failure()
            .validate_recovery_continuation(&failure_anchor)
            .is_ok()
            && error.failure().failure_id() == failure_anchor.failure_id(),
    );
    check(passed, error.failure().completed().len() == 1);
    let released = error.complete().unwrap();
    check(
        passed,
        released
            .recovery_history()
            .last()
            .unwrap()
            .validate_recovery_continuation(&failure_anchor)
            .is_ok(),
    );
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(
        passed,
        calls(&trace, "release")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
                failure_key("release", &resources[1]),
            ],
    );
}

fn full_pool_retention_and_release(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "full-pool-retention")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let deferred = committed.defer_all().unwrap();
    check(passed, deferred.entries().len() == resources.len());
    check(passed, deferred.before() == ResourceLeaseState::Active);
    check(passed, deferred.after() == ResourceLeaseState::Deferred);
    check(
        passed,
        deferred.decision() == ResourceRetentionDecision::Retain,
    );
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Deferred,
    );
    check(
        passed,
        committed
            .lease()
            .entries()
            .all(|entry| entry.state() == ResourceLeaseState::Deferred),
    );
    let resumed = committed.resume_all().unwrap();
    check(passed, resumed.before() == ResourceLeaseState::Deferred);
    check(passed, resumed.after() == ResourceLeaseState::Active);
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Active,
    );
    let cancelled = committed.cancel_all().unwrap();
    check(
        passed,
        cancelled.decision() == ResourceRetentionDecision::ReturnRequested,
    );
    check(passed, cancelled.after() == ResourceLeaseState::Cancelled);
    check(passed, cancelled.entries().len() == resources.len());
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Cancelled,
    );

    let expected_retention = plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.resource_id(),
                ResourceRetentionPolicy::from(allocation.lifetime()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    check(
        passed,
        committed.lease().entries().all(|entry| {
            expected_retention.get(entry.resource_id()) == Some(&entry.retention_policy())
        }),
    );

    let released = committed.release().unwrap();
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(
        passed,
        released.receipts().last().unwrap().records().len() == resources.len(),
    );
}

fn failure_identity_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "failure-id-sequence")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let first = committed.complete_pending_release().unwrap_err();
    let second = committed.complete_pending_release().unwrap_err();
    check(
        passed,
        first.failure_id().get() != 0 && second.failure_id().get() > first.failure_id().get(),
    );
    check(
        passed,
        first.action() == ResourceTransactionAction::Release
            && first.failure_point().is_none()
            && first.completed().is_empty()
            && first.recovery_strategy() == ResourceRecoveryStrategy::ForwardCompletion,
    );
    check(
        passed,
        second.validate_recovery_continuation(&first).is_err(),
    );
    let _released = committed.release().unwrap();
}

fn context_bound_wire_validation(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "wire-transition")
        .reserve()
        .unwrap();
    let receipt = reserved.receipts()[0].clone();
    let context = reserved
        .latest_transition_validation_context()
        .unwrap()
        .clone();
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    check(passed, wire.clone().try_validate().is_err());
    check(passed, wire.try_validate_against(&context).is_ok());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["records"][0]["generation"] = json!(0);
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["records"].as_array_mut().unwrap().pop();
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    let extra = value["records"][0].clone();
    value["records"].as_array_mut().unwrap().push(extra);
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["admission"]["plan_hash"] = json!(sha('f'));
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["identity"]["transaction_id"] = json!("transaction.mutated");
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let committed = reserved.commit().unwrap();
    let mut committed = committed;
    let lease_receipt = committed.defer_all().unwrap();
    let lease_context = committed.latest_lease_validation_context().unwrap().clone();
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&lease_receipt).unwrap(),
    )
    .unwrap();
    check(passed, wire.clone().try_validate().is_err());
    check(passed, wire.try_validate_against(&lease_context).is_ok());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    value["entries"][0]["generation"] = json!(0);
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    value["entries"].as_array_mut().unwrap().pop();
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    let extra = value["entries"][0].clone();
    value["entries"].as_array_mut().unwrap().push(extra);
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let at_limit = vec![b' '; MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES];
    let error = UnvalidatedResourceTransitionReceipt::decode_untrusted(&at_limit).unwrap_err();
    check(passed, !error.to_string().contains("exceeds limit"));
    let over_limit = vec![b' '; MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES + 1];
    let error = UnvalidatedResourceTransitionReceipt::decode_untrusted(&over_limit).unwrap_err();
    check(passed, error.to_string().contains("exceeds limit"));
    let at_limit = vec![b' '; MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES];
    let error = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(&at_limit).unwrap_err();
    check(passed, !error.to_string().contains("exceeds limit"));
    let over_limit = vec![b' '; MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES + 1];
    let error =
        UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(&over_limit).unwrap_err();
    check(passed, error.to_string().contains("exceeds limit"));

    let _terminal = committed.release().unwrap();
}

fn drop_abandon_exactly_once(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    drop(transaction(plan, driver, "drop-new"));
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0].state() == ResourceTransactionState::New,
    );
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .all(|entry| entry.transaction_state() == ResourceTransactionState::New),
    );

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let committed = transaction(plan, driver, "drop-committed")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    drop(committed);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .all(ResourceLedgerEntrySnapshot::buffer_present),
    );

    let resources = plan_resources(plan);
    let reserve_failure = failure_key("reserve", &resources[1]);
    let failures = [(reserve_failure.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let error = expect_err(transaction(plan, driver, "drop-recovery-owner").reserve());
    drop(error);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0].pending_action()
            == Some(ResourceTransactionAction::Reserve),
    );

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let terminal = transaction(plan, driver, "drop-terminal")
        .reserve()
        .unwrap()
        .rollback()
        .unwrap();
    drop(terminal);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "drop-deferred-pool")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    committed.defer_all().unwrap();
    drop(committed);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    let signal = trace.lock().unwrap().abandon[0].clone();
    check(
        passed,
        signal
            .ledger()
            .iter()
            .all(|entry| entry.transaction_state() == ResourceTransactionState::Committed),
    );
    check(
        passed,
        trace.lock().unwrap().abandon_buffer_counts == [resources.len()],
    );
}

fn abandon_callback_panic_during_unwind_is_contained(passed: &mut usize) {
    let executable = std::env::current_exe().expect("test executable is discoverable");
    let status = Command::new(executable)
        .arg("resource_transaction_abandon_panic_child")
        .arg("--exact")
        .arg("--test-threads=1")
        .arg("--nocapture")
        .env("FERRUM_VNEXT_ABANDON_PANIC_CHILD", "1")
        .env("RUST_BACKTRACE", "0")
        .status()
        .expect("abandon panic child starts");
    check(passed, status.success());
}

#[test]
fn resource_transaction_abandon_panic_child() {
    if std::env::var_os("FERRUM_VNEXT_ABANDON_PANIC_CHILD").is_none() {
        return;
    }

    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let runtime = Arc::downgrade(driver.runtime());
    trace.lock().unwrap().panic_on_abandon = true;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _committed = transaction(&plan, driver, "abandon-panic-during-unwind")
            .reserve()
            .unwrap()
            .commit()
            .unwrap();
        panic!("primary transaction owner panic");
    }));
    assert!(unwind.is_err());
    assert_eq!(trace.lock().unwrap().abandon.len(), 1);
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert!(runtime.upgrade().is_some());
}

fn allocation_withholding_is_core_owned(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-forget-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(first.to_string(), PostAllocationBehavior::ForgetThenError);
    let poisoned = match reserved.commit() {
        Err(ResourceCommitTransitionError::Poisoned(owner)) => owner,
        Err(ResourceCommitTransitionError::Recoverable(_)) => {
            panic!("withheld allocation was treated as unowned")
        }
        Ok(_) => panic!("withheld allocation unexpectedly committed"),
    };
    check(
        passed,
        poisoned.failure().failure().code() == "commit-after-allocation_failed",
    );
    check(
        passed,
        poisoned.failure().recovery_strategy() == ResourceRecoveryStrategy::ReconcileOrQuarantine,
    );
    check(
        passed,
        poisoned
            .failure()
            .ledger_after()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    check(passed, trace.lock().unwrap().runtime_allocate_calls == 1);
    let quarantined = poisoned.quarantine().unwrap();
    check(
        passed,
        quarantined.state() == ResourceTransactionState::Quarantined,
    );
    check(passed, trace.lock().unwrap().quarantine_sizes == [1]);
    drop(quarantined);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-drop-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    {
        let mut trace = trace.lock().unwrap();
        trace
            .post_allocation
            .insert(first.to_string(), PostAllocationBehavior::DropThenError);
        trace.failures.insert(failure_key("reconcile", &first), 1);
    }
    let poisoned = match reserved.commit() {
        Err(ResourceCommitTransitionError::Poisoned(owner)) => owner,
        Err(ResourceCommitTransitionError::Recoverable(_)) => {
            panic!("dropped allocation receipt lost core ownership")
        }
        Ok(_) => panic!("dropped allocation receipt unexpectedly committed"),
    };
    check(
        passed,
        poisoned.failure().failure().code() == "commit-after-allocation_failed",
    );
    let retry_owner = match poisoned.reconcile() {
        Ok(_) => panic!("injected reconcile failure unexpectedly succeeded"),
        Err(owner) => owner,
    };
    check(passed, retry_owner.failure().recovery_failures().len() == 1);
    check(
        passed,
        retry_owner
            .failure()
            .ledger_after()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    let recovery = retry_owner.reconcile().unwrap();
    check(
        passed,
        recovery.failure().recovery_strategy() == ResourceRecoveryStrategy::ReverseCompensation,
    );
    check(passed, calls(&trace, "reconcile:").len() == 2);
    let reserved = recovery.recover().unwrap();
    check(
        passed,
        reserved.state() == ResourceTransactionState::Reserved,
    );
    let rolled_back = reserved.rollback().unwrap();
    check(
        passed,
        rolled_back.state() == ResourceTransactionState::RolledBack,
    );
    drop(rolled_back);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-panic-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(first.to_string(), PostAllocationBehavior::Panic);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = reserved.commit();
    }));
    check(passed, panic_result.is_err());
    check(passed, trace.lock().unwrap().runtime_allocate_calls == 1);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(passed, trace.lock().unwrap().abandon_buffer_counts == [1]);
    check(
        passed,
        trace.lock().unwrap().abandon_claimed_bytes == [plan.payload().memory().static_bytes()],
    );
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    check(
        passed,
        admit_resources(plan, id("request.allocation-panic.after-abandon")).is_ok(),
    );
}

fn admit_logical_child_sequence(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    request: &Arc<AdmittedRequestResources<TestRuntime>>,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let admission = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match request.try_admit_sequence(admission.clone()).unwrap() {
            SequenceResourceAdmissionDecision::Admitted(resources) => return resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                root.maintain_for_deferred(&deferred).unwrap();
            }
            SequenceResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("child sequence backing did not converge after bounded maintenance")
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("child sequence unexpectedly deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("child sequence unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded child sequence admission always returns or panics")
}

fn logical_sequence_activation_completion_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-activation-completion");
    let run_id = "run.logical.activation-completion";
    let request_id = "request.logical.activation-completion";
    let resources = admit_logical_sequence(&root, run_id, request_id);
    let request_resources = resources.request_resources();
    let expected_request_slices = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.lifetime() == AllocationLifetime::Request)
        .count();
    let expected_sequence_slices = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.lifetime() == AllocationLifetime::Sequence)
        .count();

    check(passed, resources.run_id() == &id::<RunId>(run_id)); // 1
    check(
        passed,
        resources.request_id() == &id::<RequestIdentity>(request_id),
    ); // 2
    check(passed, request_resources.run_id() == resources.run_id()); // 3
    check(
        passed,
        request_resources.request_id() == resources.request_id(),
    ); // 4
    check(
        passed,
        resources.request_authority() == request_resources.request_authority(),
    ); // 5
    check(
        passed,
        resources.coordinator_id() == request_resources.coordinator_id(),
    ); // 6
    check(passed, resources.static_provisioning().is_some()); // 7
    check(
        passed,
        request_resources.backing_slices().len() == expected_request_slices,
    ); // 8
    check(
        passed,
        resources.backing_slices().len() == expected_sequence_slices,
    ); // 9

    let evidence = resources.plan_evidence();
    check(passed, evidence.plan_id() == plan.payload().plan_id()); // 10
    check(passed, evidence.plan_hash() == plan.plan_hash()); // 11
    check(passed, evidence.device_id() == plan.payload().device_id()); // 12
    check(
        passed,
        evidence.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 13

    let mut stream = resources.create_execution_stream().unwrap();
    let mut competing_stream = resources.create_execution_stream().unwrap();
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 0); // 14
    let permit = resources.activate(&mut stream).unwrap();
    let first_epoch = permit.activation_epoch();
    check(passed, std::ptr::eq(permit.resources(), resources.as_ref())); // 15
    check(passed, permit.run_id() == resources.run_id()); // 16
    check(passed, permit.request_id() == resources.request_id()); // 17
    check(
        passed,
        permit.sequence_authority() == resources.sequence_authority(),
    ); // 18
    check(
        passed,
        permit.coordinator_id() == resources.coordinator_id(),
    ); // 19
    check(
        passed,
        permit.backing_slices().len() == resources.backing_slices().len(),
    ); // 20
    check(
        passed,
        permit.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 21
    check(passed, first_epoch == 1); // 22
    check(passed, resources.activate(&mut competing_stream).is_err()); // 23

    let first_receipt = permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        first_receipt.sequence_authority() == resources.sequence_authority()
            && first_receipt.activation_epoch() == first_epoch
            && first_receipt.run_id() == resources.run_id()
            && first_receipt.request_id() == resources.request_id()
            && first_receipt.plan().plan_id() == plan.payload().plan_id()
            && first_receipt.runtime_implementation_fingerprint()
                == plan.payload().device_runtime_implementation_fingerprint()
            && trace.lock().unwrap().runtime_synchronize_calls == 1
            && !resources.is_poisoned(),
    ); // 24

    let second_permit = resources.activate(&mut stream).unwrap();
    let second_epoch = second_permit.activation_epoch();
    let second_receipt = second_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        second_epoch > first_epoch
            && second_receipt.activation_epoch() == second_epoch
            && second_receipt.sequence_authority() == resources.sequence_authority()
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !resources.is_poisoned(),
    ); // 25

    drop(competing_stream);
    drop(stream);
    drop(resources);
    let _close = close_plan_runtime(root);
}

fn logical_sequence_synchronization_retry_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_failures = 1;
    let root = plan_runtime(plan, driver, "logical-synchronization-retry");
    let resources = admit_logical_sequence(
        &root,
        "run.logical.synchronization-retry",
        "request.logical.synchronization-retry",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let epoch = permit.activation_epoch();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("injected synchronization failure unexpectedly succeeded"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Runtime(_)),
    ); // 1
    check(
        passed,
        trace.lock().unwrap().runtime_synchronize_calls == 1 && !resources.is_poisoned(),
    ); // 2
    let receipt = failure.retry().unwrap().complete().unwrap();
    check(
        passed,
        receipt.sequence_authority() == resources.sequence_authority()
            && receipt.activation_epoch() == epoch
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !resources.is_poisoned(),
    ); // 3

    drop(stream);
    drop(resources);
    let _close = close_plan_runtime(root);
}

fn deferred_admission_has_no_execution_authority(_plan: &ExecutionPlan, passed: &mut usize) {
    let constrained_plan = execution_plan_with_policy(policy_with_memory(4096, 128, 1));
    let (driver, _) = configured_driver(&constrained_plan, &[], &[]);
    let root = plan_runtime(&constrained_plan, driver, "logical-deferred-no-authority");
    let request = admit_logical_request(
        &root,
        "run.logical.deferred-no-authority",
        "request.logical.deferred-no-authority",
    );
    let first = admit_logical_child_sequence(&root, &request);
    let admission = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut deferred = None;
    for attempt in 0..=3 {
        match request.try_admit_sequence(admission.clone()).unwrap() {
            SequenceResourceAdmissionDecision::Deferred(decision) => {
                deferred = Some(decision);
                break;
            }
            SequenceResourceAdmissionDecision::BackingDeferred(backing) if attempt < 3 => {
                root.maintain_for_deferred(&backing).unwrap();
            }
            SequenceResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("deferred admission backing did not converge after bounded maintenance")
            }
            SequenceResourceAdmissionDecision::Admitted(_) => {
                panic!("sequence ceiling produced executable authority")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("sequence ceiling was treated as permanent rejection")
            }
        }
    }
    let deferred = deferred.expect("bounded admission reaches logical deferral");
    // The Deferred variant carries only retry evidence. In particular, this
    // branch never yields Arc<AdmittedSequenceResources<_>>, so no stream or
    // ActiveSequencePermit can be minted from the rejected attempt.
    check(
        passed,
        deferred.action() == DeferredAction::WaitForRelease
            && deferred.available().active_sequences() == 1
            && deferred.available().maximum_active_sequences() == 1
            && deferred.blockers().iter().any(|blocker| {
                blocker.kind() == CapacityShortfallKind::ActiveSequenceCeiling
                    && blocker.domain().is_none()
            }),
    ); // 1

    drop(deferred);
    drop(first);
    drop(request);
    let _close = close_plan_runtime(root);
}

fn logical_sequence_explicit_abort_is_exact_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-explicit-abort-exact");
    let run_id = "run.logical.explicit-abort-exact";
    let request_id = "request.logical.explicit-abort-exact";
    let request = admit_logical_request(&root, run_id, request_id);
    let aborted_resources = admit_logical_child_sequence(&root, &request);
    let sibling_resources = admit_logical_child_sequence(&root, &request);

    check(
        passed,
        aborted_resources.request_authority() == sibling_resources.request_authority(),
    ); // 1
    check(
        passed,
        aborted_resources.sequence_authority() != sibling_resources.sequence_authority(),
    ); // 2
    check(
        passed,
        aborted_resources.coordinator_id() == sibling_resources.coordinator_id(),
    ); // 3
    check(
        passed,
        Arc::ptr_eq(
            aborted_resources.request_resources(),
            sibling_resources.request_resources(),
        ),
    ); // 4
    check(
        passed,
        !aborted_resources.is_poisoned() && !sibling_resources.is_poisoned(),
    ); // 5

    let mut aborted_stream = aborted_resources.create_execution_stream().unwrap();
    let mut sibling_stream = sibling_resources.create_execution_stream().unwrap();
    let aborted_permit = aborted_resources.activate(&mut aborted_stream).unwrap();
    let sibling_permit = sibling_resources.activate(&mut sibling_stream).unwrap();
    let aborted_epoch = aborted_permit.activation_epoch();
    let sibling_epoch = sibling_permit.activation_epoch();
    check(
        passed,
        aborted_permit.sequence_authority() == aborted_resources.sequence_authority(),
    ); // 6
    check(
        passed,
        sibling_permit.sequence_authority() == sibling_resources.sequence_authority(),
    ); // 7
    check(passed, aborted_epoch == 1 && sibling_epoch == 1); // 8
    check(
        passed,
        aborted_permit.sequence_authority() != sibling_permit.sequence_authority(),
    ); // 9

    let abort = aborted_permit.synchronize().unwrap().abort().unwrap();
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 10
    check(
        passed,
        abort.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 11
    check(
        passed,
        abort.sequence_authority() == aborted_resources.sequence_authority(),
    ); // 12
    check(passed, abort.run_id() == &id::<RunId>(run_id)); // 13
    check(
        passed,
        abort.request_id() == &id::<RequestIdentity>(request_id),
    ); // 14
    check(passed, abort.activation_epoch() == aborted_epoch); // 15
    check(
        passed,
        abort.plan().coordinator_id() == aborted_resources.coordinator_id(),
    ); // 16
    check(
        passed,
        abort.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 17
    check(passed, aborted_resources.is_poisoned()); // 18
    check(
        passed,
        matches!(
            aborted_resources.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 19
    check(
        passed,
        aborted_resources.activate(&mut aborted_stream).is_err(),
    ); // 20
    check(passed, !sibling_resources.is_poisoned()); // 21

    let sibling_completion = sibling_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        sibling_completion.sequence_authority() == sibling_resources.sequence_authority()
            && sibling_completion.activation_epoch() == sibling_epoch
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !sibling_resources.is_poisoned(),
    ); // 22

    let third_resources = admit_logical_child_sequence(&root, &request);
    let mut third_stream = third_resources.create_execution_stream().unwrap();
    let third_permit = third_resources.activate(&mut third_stream).unwrap();
    let third_completion = third_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        third_completion.sequence_authority() == third_resources.sequence_authority()
            && third_resources.sequence_authority() != aborted_resources.sequence_authority()
            && !third_resources.is_poisoned()
            && trace.lock().unwrap().runtime_synchronize_calls == 3,
    ); // 23

    drop(third_stream);
    drop(third_resources);
    drop(sibling_stream);
    drop(sibling_resources);
    drop(aborted_stream);
    drop(aborted_resources);
    drop(request);
    let close = close_plan_runtime(root);
    let trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && trace.quarantine_sizes.is_empty()
            && trace.abandon.is_empty(),
    ); // 24
}

fn logical_sequence_drop_recovery_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-drop-recovery");
    let run_id = "run.logical.drop-recovery";
    let request_id = "request.logical.drop-recovery";
    let request = admit_logical_request(&root, run_id, request_id);
    let resources = admit_logical_child_sequence(&root, &request);
    let sibling_resources = admit_logical_child_sequence(&root, &request);
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    drop(permit);

    check(passed, resources.is_poisoned()); // 1
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 0); // 2
    check(
        passed,
        matches!(
            resources.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == resources.sequence_authority().sparse_id()
                && epoch == activation_epoch
        ),
    ); // 3

    // Dropping the exact bound stream transfers it into the private recovery
    // registry. Recovery then drains and retires only this sequence authority.
    drop(stream);
    let recovered = resources.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 4
    check(
        passed,
        recovered.sequence_authority() == resources.sequence_authority(),
    ); // 5
    check(passed, recovered.activation_epoch() == activation_epoch); // 6
    check(passed, recovered.run_id() == &id::<RunId>(run_id)); // 7
    check(
        passed,
        recovered.request_id() == &id::<RequestIdentity>(request_id),
    ); // 8
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 9
    check(
        passed,
        matches!(
            resources.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ) && matches!(
            resources.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 10

    let mut sibling_stream = sibling_resources.create_execution_stream().unwrap();
    let sibling_permit = sibling_resources.activate(&mut sibling_stream).unwrap();
    let sibling_completion = sibling_permit.synchronize().unwrap().complete().unwrap();
    drop(sibling_stream);
    drop(sibling_resources);
    drop(resources);
    drop(request);
    let close = close_plan_runtime(root);
    let trace = trace.lock().unwrap();
    check(
        passed,
        sibling_completion.sequence_authority() != recovered.sequence_authority()
            && trace.runtime_synchronize_calls == 2
            && close.released_static_resources() == plan_resources(plan).len()
            && trace.quarantine_sizes.is_empty()
            && trace.abandon.is_empty()
            && trace.stream_drops == 2,
    ); // 11
}

const ABANDONED_RECOVERY_CONCURRENT_WORKERS: usize = 1;
const _: () = assert!(
    ABANDONED_RECOVERY_CONCURRENT_WORKERS == 1
        && ABANDONED_RECOVERY_CONCURRENT_WORKERS <= MAX_RESOURCE_TEST_CONCURRENT_WORKERS
);

fn forgotten_live_permit_fails_closed(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "forgotten-logical-permit");
    let sequence = admit_logical_sequence(
        &root,
        "run.forgotten-logical-permit",
        "request.forgotten-logical-permit",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    check(passed, activation_epoch != 0); // 1/16

    // This is intentionally the hostile path: Drop cannot mark the sequence
    // poisoned, so the registry must retain exact epoch metadata and refuse
    // recovery until the externally-owned bound stream is returned.
    std::mem::forget(permit);
    check(passed, !sequence.is_poisoned()); // 2/16
    check(passed, sequence.activate(&mut stream).is_err()); // 3/16
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == sequence_authority.sparse_id() && epoch == activation_epoch
        ),
    ); // 4/16

    // The logical sequence owns a request binding, which owns the root Arc.
    // Close therefore enters Closing but cannot tear down resources underneath
    // the forgotten permit's recovery authority.
    let (root, strong_count) = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            ..
        }) => (resources, strong_count),
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("forgotten logical permit unexpectedly allowed root close")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    check(passed, strong_count >= 2); // 5/16
    check(passed, root.is_closing()); // 6/16
    check(passed, root.trusted_runtime_binding().is_err()); // 7/16

    drop(stream);
    check(passed, sequence.is_poisoned()); // 8/16
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(passed, recovered.sequence_authority() == sequence_authority); // 9/16
    check(passed, recovered.activation_epoch() == activation_epoch); // 10/16
    check(passed, recovered.run_id() == sequence.run_id()); // 11/16
    check(passed, recovered.request_id() == sequence.request_id()); // 12/16
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
            && recovered.runtime_implementation_fingerprint()
                == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 13/16
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 14/16
    check(
        passed,
        matches!(
            sequence.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 15/16

    drop(sequence);
    let close = close_plan_runtime(root);
    let final_trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && final_trace.runtime_synchronize_calls == 1
            && final_trace.stream_drops == 1
            && final_trace.abandon.is_empty(),
    ); // 16/16
}

fn abandoned_sequence_recovery_contract(plan: &ExecutionPlan, passed: &mut usize) {
    // Backend synchronization returning Ok is insufficient unless the exact
    // stream also reports Ready. The failure owner is dropped intentionally so
    // the registry must recover the abandoned stream.
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_returns_not_ready = true;
    let root = plan_runtime(plan, driver, "recovery-not-ready");
    let sequence = admit_logical_sequence(
        &root,
        "run.recovery-not-ready",
        "request.recovery-not-ready",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("non-ready stream produced synchronization evidence"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Contract(_)),
    ); // 1/14
    drop(failure);
    check(passed, sequence.is_poisoned()); // 2/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == sequence_authority.sparse_id() && epoch == activation_epoch
        ),
    ); // 3/14
    drop(stream);
    trace.lock().unwrap().synchronize_returns_not_ready = false;
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(passed, recovered.sequence_authority() == sequence_authority); // 4/14
    check(
        passed,
        recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 5/14
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 2); // 6/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 7/14
    drop(sequence);
    let close = close_plan_runtime(root);
    let final_trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && final_trace.stream_drops == 1
            && final_trace.abandon.is_empty(),
    ); // 8/14
    drop(final_trace);

    // A backend error must restore the attached stream to the registry so a
    // later bounded retry can drain it. Two failures cover the explicit permit
    // path and the first registry recovery attempt independently.
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_failures = 2;
    let root = plan_runtime(plan, driver, "recovery-runtime-retry");
    let sequence = admit_logical_sequence(
        &root,
        "run.recovery-runtime-retry",
        "request.recovery-runtime-retry",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("injected synchronization failure unexpectedly succeeded"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Runtime(_)),
    ); // 9/14
    drop(failure);
    drop(stream);
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Runtime(_))
        ),
    ); // 10/14
    check(
        passed,
        sequence.is_poisoned() && trace.lock().unwrap().runtime_synchronize_calls == 2,
    ); // 11/14
    trace.lock().unwrap().synchronize_failures = 0;
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.sequence_authority() == sequence_authority
            && recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 12/14
    let (runtime_synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    check(passed, runtime_synchronize_calls == 3 && stream_drops == 1); // 13/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 14/14
    drop(sequence);
    let _ = close_plan_runtime(root);
}

fn abandoned_recovery_unlocks_during_runtime_sync(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "concurrent-abandoned-recovery");
    let sequence = admit_logical_sequence(
        &root,
        "run.concurrent-abandoned-recovery",
        "request.concurrent-abandoned-recovery",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    drop(permit);
    drop(stream);
    check(passed, sequence.is_poisoned()); // 1/7

    let entered = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    let release = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    trace.lock().unwrap().synchronize_block = Some((Arc::clone(&entered), Arc::clone(&release)));

    // Exactly one worker performs the slow recovery; the caller is the only
    // contender. Spawn is completed before either barrier wait. After the
    // entered rendezvous, no assertion or fallible unwrap occurs until the
    // release rendezvous and join have completed, so a failed assertion cannot
    // strand the already-started worker on an unreachable barrier.
    let worker_sequence = Arc::clone(&sequence);
    let (concurrent, worker_result, calls_while_blocked) = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("vnext-abandoned-recovery-worker".to_owned())
            .spawn_scoped(scope, move || worker_sequence.recover_abandoned_sequence())
            .expect("the single bounded abandoned-recovery worker starts");
        entered.wait();
        let calls_while_blocked = trace.lock().unwrap().runtime_synchronize_calls;
        let concurrent = sequence.recover_abandoned_sequence();
        release.wait();
        let worker_result = worker
            .join()
            .expect("the bounded abandoned-recovery worker does not panic");
        (concurrent, worker_result, calls_while_blocked)
    });
    let recovered = match worker_result {
        Ok(receipt) => receipt,
        Err(_) => panic!("the primary abandoned recovery unexpectedly failed"),
    };

    check(passed, calls_while_blocked == 1); // 2/7
    check(
        passed,
        matches!(concurrent, Err(AbandonedSequenceRecoveryError::Contract(_))),
    ); // 3/7
    check(passed, recovered.sequence_authority() == sequence_authority); // 4/7
    check(passed, recovered.activation_epoch() == activation_epoch); // 5/7
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
            && trace.lock().unwrap().runtime_synchronize_calls == 1,
    ); // 6/7
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ) && trace.lock().unwrap().stream_drops == 1,
    ); // 7/7

    drop(sequence);
    let _ = close_plan_runtime(root);
}

fn abandoned_recovery_retains_stream_on_backend_panic(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "abandoned-recovery-backend-panic");
    let sequence = admit_logical_sequence(
        &root,
        "run.abandoned-recovery-backend-panic",
        "request.abandoned-recovery-backend-panic",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    drop(permit);
    drop(stream);
    trace.lock().unwrap().panic_on_stream_state = true;

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = sequence.recover_abandoned_sequence();
    }));
    check(passed, panic.is_err()); // 1/5
    check(passed, sequence.is_poisoned()); // 2/5
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 3/5

    // recover() restores the stream to Attached before resuming the backend
    // panic, so the second attempt must recover the same epoch rather than
    // leaking or replacing the raw stream.
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.sequence_authority() == sequence_authority
            && recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 4/5
    let (runtime_synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    check(
        passed,
        runtime_synchronize_calls == 2
            && stream_drops == 1
            && matches!(
                sequence.recover_abandoned_sequence(),
                Err(AbandonedSequenceRecoveryError::Contract(_))
            ),
    ); // 5/5

    drop(sequence);
    let _ = close_plan_runtime(root);
}

fn unresolved_owner_drop_retains_backend_lifetimes(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let second_release = failure_key("release", &resources[1]);
    let (driver, trace) = configured_driver(plan, &[(second_release.as_str(), 1)], &[]);
    let runtime = Arc::downgrade(driver.runtime());
    trace.lock().unwrap().retain_ownership = true;

    let root = plan_runtime(plan, driver, "unresolved-close-owner-drop");
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected plan runtime close failure unexpectedly succeeded"),
    };
    check(passed, close_failure.failure().code() == "release_failed"); // C1

    let drops_before_owner_drop = trace.lock().unwrap().buffer_drops;
    drop(close_failure);
    let owner_drop_recorded = {
        let trace = trace.lock().unwrap();
        trace.abandon.len() == 1 && trace.durable_ownership.len() == 1
    };
    check(passed, owner_drop_recorded); // C2

    let ownership = trace
        .lock()
        .unwrap()
        .durable_ownership
        .pop()
        .expect("failed close drop transfers unresolved ownership to the driver");
    check(
        passed,
        ownership.reason() == ResourceOwnershipReason::Abandon,
    ); // C3
    check(
        passed,
        ownership.buffers().len() == 1
            && ownership.claimed_bytes()
                == ownership
                    .buffers()
                    .iter()
                    .map(|buffer| buffer.expected_descriptor().size_bytes)
                    .sum::<u64>(),
    ); // C4
    check(
        passed,
        runtime.upgrade().is_some()
            && trace.lock().unwrap().buffer_drops == drops_before_owner_drop,
    ); // C5

    drop(ownership);
    let (buffer_drops, buffer_drops_after_backend) = {
        let trace = trace.lock().unwrap();
        (trace.buffer_drops, trace.buffer_drops_after_backend)
    };
    check(
        passed,
        runtime.upgrade().is_none()
            && buffer_drops == drops_before_owner_drop + 1
            && buffer_drops_after_backend == 0,
    ); // C6
}

fn abort_receipt_authority_and_cleanup_retry_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);

    let (source_driver, source_trace) = configured_driver(plan, &[], &[]);
    let source_root = plan_runtime(plan, source_driver, "logical-abort-source");
    let source_run = "run.logical-abort.source";
    let source_request = "request.logical-abort.source";
    let source_sequence = admit_logical_sequence(&source_root, source_run, source_request);
    let source_coordinator = source_sequence.coordinator_id();
    let source_authority = source_sequence.sequence_authority();
    let mut source_stream = source_sequence.create_execution_stream().unwrap();
    let source_active = source_sequence.activate(&mut source_stream).unwrap();
    let source_epoch = source_active.activation_epoch();
    let source_abort = source_active.synchronize().unwrap().abort().unwrap();

    check(
        passed,
        source_abort.plan().coordinator_id() == source_coordinator,
    ); // C1
    check(
        passed,
        source_abort.sequence_authority() == source_authority,
    ); // C2
    check(passed, source_abort.run_id().as_str() == source_run); // C3
    check(passed, source_abort.request_id().as_str() == source_request); // C4
    check(passed, source_abort.activation_epoch() == source_epoch); // C5
    check(
        passed,
        source_abort.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C6
    check(
        passed,
        source_abort.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // C7

    let second_release = failure_key("release", &resources[1]);
    let (target_driver, target_trace) =
        configured_driver(plan, &[(second_release.as_str(), 1)], &[]);
    let target_root = plan_runtime(plan, target_driver, "logical-abort-target");
    let target_run = "run.logical-abort.target";
    let target_request = "request.logical-abort.target";
    let target_sequence = admit_logical_sequence(&target_root, target_run, target_request);
    let target_coordinator = target_sequence.coordinator_id();
    let target_authority = target_sequence.sequence_authority();
    check(
        passed,
        source_abort.plan().coordinator_id() != target_coordinator
            && (
                source_abort.plan().coordinator_id(),
                source_abort.sequence_authority(),
            ) != (target_coordinator, target_authority),
    ); // C8

    drop(source_sequence);
    drop(source_stream);
    let source_close = close_plan_runtime(source_root);
    let source_cleanup_is_clean = {
        let trace = source_trace.lock().unwrap();
        trace.quarantine_sizes.is_empty() && trace.abandon.is_empty()
    };
    check(
        passed,
        source_close.released_static_resources() == resources.len() && source_cleanup_is_clean,
    ); // C9

    let mut target_stream = target_sequence.create_execution_stream().unwrap();
    let target_active = target_sequence.activate(&mut target_stream).unwrap();
    let target_abort = target_active.synchronize().unwrap().abort().unwrap();
    check(
        passed,
        target_abort.plan().coordinator_id() == target_coordinator
            && target_abort.sequence_authority() == target_authority
            && target_abort.plan().coordinator_id() != source_abort.plan().coordinator_id(),
    ); // C10

    drop(target_sequence);
    drop(target_stream);
    let close_failure = match PlanRuntimeResources::close(target_root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected target cleanup failure unexpectedly succeeded"),
    };
    check(
        passed,
        close_failure.failure().code() == "release_failed"
            && target_trace.lock().unwrap().quarantine_sizes.is_empty(),
    ); // C11
    let target_close = match close_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("target cleanup retry unexpectedly failed"),
    };
    let target_cleanup_is_clean = {
        let trace = target_trace.lock().unwrap();
        trace.quarantine_sizes.is_empty() && trace.abandon.is_empty()
    };
    check(
        passed,
        target_close.released_static_resources() == resources.len() && target_cleanup_is_clean,
    ); // C12
}

fn bound_stream_sequence_authority_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "bound-logical-sequence-stream");
    let source = admit_logical_sequence(
        &root,
        "run.bound-sequence.source",
        "request.bound-sequence.source",
    );
    let target = admit_logical_sequence(
        &root,
        "run.bound-sequence.target",
        "request.bound-sequence.target",
    );

    check(
        passed,
        source.coordinator_id() == target.coordinator_id()
            && source.sequence_authority() != target.sequence_authority(),
    ); // C1
    check(
        passed,
        source.request_authority() != target.request_authority(),
    ); // C2

    let mut source_stream = source.create_execution_stream().unwrap();
    check(passed, target.activate(&mut source_stream).is_err()); // C3
    check(passed, !target.is_poisoned()); // C4

    let source_active = source.activate(&mut source_stream).unwrap();
    check(
        passed,
        source_active.sequence_authority() == source.sequence_authority()
            && source_active.coordinator_id() == source.coordinator_id(),
    ); // C5
    let source_completion = source_active.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        source_completion.sequence_authority() == source.sequence_authority()
            && source_completion.plan().coordinator_id() == source.coordinator_id(),
    ); // C6

    check(
        passed,
        target.activate(&mut source_stream).is_err() && !target.is_poisoned(),
    ); // C7

    let mut target_stream = target.create_execution_stream().unwrap();
    let target_completion = target
        .activate(&mut target_stream)
        .unwrap()
        .synchronize()
        .unwrap()
        .complete()
        .unwrap();
    check(
        passed,
        target_completion.sequence_authority() == target.sequence_authority()
            && target_completion.plan().coordinator_id() == target.coordinator_id(),
    ); // C8

    drop(source);
    drop(target);
    drop(source_stream);
    drop(target_stream);
    drop(source_completion);
    drop(target_completion);
    let _ = close_plan_runtime(root);
}

// Call this once from `runtime_implementation_authority_contract`, or call it
// as a sibling from the exhaustive test. It contributes exactly six checks.
fn runtime_implementation_authority_root_extension(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "runtime-authority-root");
    let binding = root.trusted_runtime_binding().unwrap();
    check(
        passed,
        binding.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C1
    check(
        passed,
        binding.evidence().runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C2

    let request = admit_logical_request(
        &root,
        "run.runtime-authority.request",
        "request.runtime-authority.request",
    );
    check(
        passed,
        request.plan_evidence().runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C3

    let sequence = admit_logical_sequence(
        &root,
        "run.runtime-authority.sequence",
        "request.runtime-authority.sequence",
    );
    check(
        passed,
        sequence
            .plan_evidence()
            .runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C4

    let mut stream = sequence.create_execution_stream().unwrap();
    let active = sequence.activate(&mut stream).unwrap();
    check(
        passed,
        active.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C5
    let completion = active.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        completion.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint()
            && completion.plan().coordinator_id() == sequence.coordinator_id(),
    ); // C6

    drop(binding);
    drop(request);
    drop(sequence);
    drop(stream);
    drop(completion);
    let _ = close_plan_runtime(root);
}

fn deferred_static_cannot_mint_binding(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "deferred-root-handoff")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    committed.defer_all().unwrap();
    let handoff_failure = match committed.into_plan_runtime() {
        Err(failure) => failure,
        Ok(_) => panic!("deferred static resources minted an owning runtime root"),
    };
    check(
        passed,
        matches!(
            handoff_failure.error(),
            VNextError::InvalidExecutionPlan { reason }
                if reason == "plan runtime handoff requires one complete active committed transaction"
        ),
    ); // C1

    let mut committed = handoff_failure.into_transaction();
    committed.resume_all().unwrap();
    let _ = committed.release().unwrap();
}

#[test]
fn plan_runtime_close_recovery_is_ownership_safe() {
    const EXPECTED_CLOSE_CASES: usize = 18;
    let plan = execution_plan();
    let resources = plan_resources(&plan);
    assert_eq!(resources.len(), 2);
    let second_release = failure_key("release", &resources[1]);
    let mut passed = 0;

    let (driver, trace) = configured_driver(&plan, &[(second_release.as_str(), 1)], &[]);
    let committed = transaction(&plan, driver, "plan-close-retry")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        close_failure.failure().code() == "release_failed",
    );
    check(
        &mut passed,
        calls(&trace, "release:")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
            ],
    );
    let retry_receipt = match close_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("plan runtime close retry unexpectedly failed"),
    };
    check(
        &mut passed,
        retry_receipt.released_static_resources() == resources.len(),
    );
    check(
        &mut passed,
        trace.lock().unwrap().quarantine_sizes.is_empty(),
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(&plan, &[(second_release.as_str(), 1)], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "plan-close-quarantine")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        close_failure.failure().code() == "release_failed",
    );
    let quarantine_receipt = match close_failure.quarantine() {
        Ok(receipt) => receipt,
        Err(_) => panic!("plan runtime quarantine unexpectedly failed"),
    };
    check(
        &mut passed,
        quarantine_receipt.released_static_resources() == 1,
    );
    check(
        &mut passed,
        quarantine_receipt.quarantined_static_resources() == 1,
    );
    check(&mut passed, trace.lock().unwrap().quarantine_sizes == [1]);
    check(
        &mut passed,
        trace.lock().unwrap().quarantine_actual_mismatch == [false],
    );
    check(
        &mut passed,
        trace.lock().unwrap().durable_ownership.len() == 1,
    );
    check(
        &mut passed,
        trace.lock().unwrap().durable_ownership[0].buffers().len() == 1,
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());
    let retained_ownership = {
        let mut trace = trace.lock().unwrap();
        std::mem::take(&mut trace.durable_ownership)
    };
    drop(retained_ownership);

    let failures = [(second_release.as_str(), 1), ("quarantine", 1)];
    let (driver, trace) = configured_driver(&plan, &failures, &[]);
    let committed = transaction(&plan, driver, "plan-close-quarantine-retry")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    let quarantine_failure = match close_failure.quarantine() {
        Err(failure) => failure,
        Ok(_) => panic!("injected quarantine failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        quarantine_failure.failure().code() == "quarantine_failed",
    );
    check(&mut passed, trace.lock().unwrap().quarantine_sizes == [1]);
    let retry_receipt = match quarantine_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("close retry after quarantine failure unexpectedly failed"),
    };
    check(
        &mut passed,
        retry_receipt.released_static_resources() == resources.len(),
    );
    check(
        &mut passed,
        calls(&trace, "release:")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
                failure_key("release", &resources[1]),
            ],
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());

    assert_eq!(passed, EXPECTED_CLOSE_CASES);
    println!("VNEXT PLAN RUNTIME CLOSE PASS: {passed}/{EXPECTED_CLOSE_CASES}");
}

fn closing_error(error: &VNextError) -> bool {
    error.to_string().contains("closing plan runtime")
}

fn begin_single_participant_step(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                root.maintain_for_deferred(&deferred).unwrap();
            }
            StepResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("step backing did not converge after bounded maintenance")
            }
            StepResourceAdmissionDecision::Deferred(_) => {
                panic!("single-participant step unexpectedly deferred")
            }
            StepResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("single-participant step unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded step admission loop always returns or panics")
}

#[test]
fn poisoned_bound_stream_retains_sequence_until_stream_drop() {
    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "poisoned-stream-sequence-hold");
    let sequence = admit_logical_sequence(
        &root,
        "run.poisoned-stream-sequence-hold",
        "request.poisoned-stream-sequence-hold",
    );
    let weak_sequence = Arc::downgrade(&sequence);
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    drop(permit);
    assert!(sequence.is_poisoned());

    drop(sequence);
    assert!(weak_sequence.upgrade().is_some());
    assert_eq!(trace.lock().unwrap().stream_drops, 0);
    let root = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("poisoned bound stream released its sequence/root hold too early")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };

    trace.lock().unwrap().synchronize_failures = 1;
    drop(stream);
    assert!(weak_sequence.upgrade().is_none());
    let pending = root.deferred_cleanup_status();
    assert_eq!(pending.pending(), 1);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);
    let first_cleanup = root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first_cleanup.retryable(), 1);
    assert_eq!(first_cleanup.status_after().pending(), 1);
    assert_eq!(trace.lock().unwrap().stream_drops, 0);
    trace.lock().unwrap().synchronize_failures = 0;
    let second_cleanup = root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(second_cleanup.completed(), 1);
    assert_eq!(second_cleanup.status_after().pending(), 0);
    let (synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    assert_eq!(synchronize_calls, 2);
    assert_eq!(stream_drops, 1);
    let _ = close_plan_runtime(root);
}

#[test]
fn sequence_owner_drop_defers_blocking_backend_recovery() {
    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "blocking-sequence-owner-drop");
    let sequence = admit_logical_sequence(
        &root,
        "run.blocking-sequence-owner-drop",
        "request.blocking-sequence-owner-drop",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    drop(permit);

    let entered = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    let release = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    trace.lock().unwrap().synchronize_block = Some((Arc::clone(&entered), Arc::clone(&release)));
    drop(sequence);

    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_returned = std::thread::scope(|scope| {
        let drop_worker = std::thread::Builder::new()
            .name("vnext-blocking-sequence-drop".to_owned())
            .spawn_scoped(scope, move || {
                drop(stream);
                let _ = dropped_tx.send(());
            })
            .expect("the single bounded sequence-drop worker starts");
        let drop_returned = dropped_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        drop_worker
            .join()
            .expect("the bounded sequence-drop worker does not panic");
        drop_returned
    });
    assert!(drop_returned, "sequence Drop waited on the backend");
    assert_eq!(root.deferred_cleanup_status().pending(), 1);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);

    let close = PlanRuntimeResources::close(root);

    let (root, strong_count, deferred_cleanup) = match close {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            deferred_cleanup,
        }) => (resources, strong_count, deferred_cleanup),
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("blocked cleanup released its plan root before quiescence")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    assert!(
        strong_count >= 2,
        "cleanup registry did not retain the plan root"
    );
    assert_eq!(deferred_cleanup.pending(), 1);

    let maintenance_root = Arc::clone(&root);
    let (cleanup, calls_while_blocked, stream_drops_while_blocked) = std::thread::scope(|scope| {
        let cleanup_worker = std::thread::Builder::new()
            .name("vnext-sequence-cleanup-recovery".to_owned())
            .spawn_scoped(scope, move || {
                maintenance_root.maintain_deferred_cleanups(1)
            })
            .expect("the single bounded sequence cleanup worker starts");
        entered.wait();
        let (calls_while_blocked, stream_drops_while_blocked) = {
            let trace = trace.lock().unwrap();
            (trace.runtime_synchronize_calls, trace.stream_drops)
        };
        release.wait();
        let cleanup = cleanup_worker
            .join()
            .expect("the bounded sequence cleanup worker does not panic")
            .expect("the bounded sequence cleanup call is valid");
        (cleanup, calls_while_blocked, stream_drops_while_blocked)
    });
    assert_eq!(calls_while_blocked, 1);
    assert_eq!(stream_drops_while_blocked, 0);
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    assert_eq!(Arc::strong_count(&root), 1);
    let final_trace = trace.lock().unwrap();
    assert_eq!(final_trace.runtime_synchronize_calls, 1);
    assert_eq!(final_trace.stream_drops, 1);
    drop(final_trace);
    let _ = close_plan_runtime(root);
}

#[test]
fn closing_root_rejects_every_parent_to_child_derivation() {
    let plan = execution_plan();
    let (driver, _trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "closing-child-derivation");
    let sequence = admit_logical_sequence(
        &root,
        "run.closing-sequence-child",
        "request.closing-sequence-child",
    );
    let request = Arc::clone(sequence.request_resources());
    let mut existing_stream = sequence.create_execution_stream().unwrap();
    let session = sequence.open_session().unwrap();
    let existing_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let existing_step = begin_single_participant_step(&root, &existing_batch);

    let root = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("live resource parents unexpectedly allowed root close")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    assert!(root.is_closing());

    let sequence_request = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        request.try_admit_sequence(sequence_request),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.open_session(),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.create_execution_stream(),
        Err(ExecutionStreamCreationError::Contract(error)) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.activate(&mut existing_stream),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        ExecutionBatchParticipants::new(vec![Arc::clone(&session)]),
        Err(error) if closing_error(&error)
    ));

    let step_request = StepResourceAdmissionRequest::new(
        existing_batch
            .bind_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        existing_batch.try_begin_step(step_request),
        Err(error) if closing_error(&error)
    ));
    let invocation_request = InvocationResourceAdmissionRequest::for_all_step_participants(
        id("node.main"),
        existing_step
            .bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        existing_step.try_admit_invocation(invocation_request),
        Err(error) if closing_error(&error)
    ));

    drop(existing_stream);
    drop(existing_step);
    drop(existing_batch);
    drop(session);
    drop(sequence);
    drop(request);
    let _ = close_plan_runtime(root);
}

#[test]
fn resource_transaction_contract_is_exhaustive() {
    let plan = execution_plan();
    let mut passed = 0;
    runtime_implementation_authority_contract(&plan, &mut passed);
    runtime_implementation_authority_root_extension(&plan, &mut passed);
    device_global_capacity_contract(&plan, &mut passed);
    logical_sequence_activation_completion_contract(&plan, &mut passed);
    logical_sequence_synchronization_retry_contract(&plan, &mut passed);
    deferred_admission_has_no_execution_authority(&plan, &mut passed);
    logical_sequence_explicit_abort_is_exact_contract(&plan, &mut passed);
    logical_sequence_drop_recovery_contract(&plan, &mut passed);
    forgotten_live_permit_fails_closed(&plan, &mut passed);
    abandoned_sequence_recovery_contract(&plan, &mut passed);
    abandoned_recovery_unlocks_during_runtime_sync(&plan, &mut passed);
    abandoned_recovery_retains_stream_on_backend_panic(&plan, &mut passed);
    unresolved_owner_drop_retains_backend_lifetimes(&plan, &mut passed);
    abort_receipt_authority_and_cleanup_retry_contract(&plan, &mut passed);
    bound_stream_sequence_authority_contract(&plan, &mut passed);
    deferred_static_cannot_mint_binding(&plan, &mut passed);
    admission_and_success(&plan, &mut passed);
    reverse_incremental_recovery(&plan, &mut passed);
    poison_reconcile_and_quarantine(&plan, &mut passed);
    forward_only_recovery(&plan, &mut passed);
    full_pool_retention_and_release(&plan, &mut passed);
    failure_identity_contract(&plan, &mut passed);
    context_bound_wire_validation(&plan, &mut passed);
    drop_abandon_exactly_once(&plan, &mut passed);
    abandon_callback_panic_during_unwind_is_contained(&mut passed);
    allocation_withholding_is_core_owned(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("VNEXT RESOURCE TRANSACTION PASS: {passed}/{EXPECTED_CASES}");
}
