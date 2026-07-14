#![allow(dead_code, unused_imports)]

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

pub(crate) const DEVICE_GLOBAL_CAPACITY_CASES: usize = 20;
pub(crate) const RESOURCE_CAPACITY_TEST_MAXIMUM_CLAIMS: usize = 16;
pub(crate) const RESOURCE_CAPACITY_CONCURRENT_WORKERS: usize = 1;
pub(crate) const MAX_RESOURCE_TEST_CONCURRENT_WORKERS: usize = 4;
pub(crate) const ABANDONED_RECOVERY_CONCURRENT_WORKERS: usize = 1;

const _: () = assert!(
    RESOURCE_CAPACITY_CONCURRENT_WORKERS == 1
        && RESOURCE_CAPACITY_CONCURRENT_WORKERS <= MAX_RESOURCE_TEST_CONCURRENT_WORKERS
);

const _: () = assert!(
    ABANDONED_RECOVERY_CONCURRENT_WORKERS == 1
        && ABANDONED_RECOVERY_CONCURRENT_WORKERS <= MAX_RESOURCE_TEST_CONCURRENT_WORKERS
);

pub(crate) fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

pub(crate) fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

pub(crate) fn one_token_span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[1], 0..1).unwrap()
}

pub(crate) fn one_token_work() -> ResourceWorkShape {
    ResourceWorkShape::single(one_token_span()).unwrap()
}

pub(crate) fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

pub(crate) fn contiguous_storage_bindings(
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
pub(crate) struct TestConfig {
    pub(crate) width: u64,
}

#[derive(Default)]
pub(crate) struct TestFamily;

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

pub(crate) fn tensor_contract(
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

pub(crate) fn operation() -> OperationDescriptor {
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

pub(crate) fn catalog() -> CapabilityCatalog {
    catalog_with_runtime_fingerprint(sha('d'))
}

pub(crate) fn catalog_with_runtime_fingerprint(
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

pub(crate) struct TestOperationContract {
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

pub(crate) struct TestEstimator {
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

pub(crate) fn operation_registry(
    catalog: &CapabilityCatalog,
) -> OperationRuntimeRegistry<TestRuntime> {
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

pub(crate) fn policy() -> ResolvedRuntimePolicy {
    policy_with_memory(4096, 128, 3)
}

pub(crate) fn policy_with_memory(
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

pub(crate) fn policy_with_memory_id(
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
            maximum_scheduled_tokens: 4096,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
    )
    .unwrap()
}

pub(crate) fn resolved_tensor(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn binding(
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

pub(crate) fn node_resolution(
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

pub(crate) fn execution_plan() -> ExecutionPlan {
    execution_plan_with_policy(policy())
}

pub(crate) fn execution_plan_with_policy(policy: ResolvedRuntimePolicy) -> ExecutionPlan {
    execution_plan_with_policy_and_runtime_fingerprint(policy, sha('d'))
}

pub(crate) fn execution_plan_with_policy_and_runtime_fingerprint(
    policy: ResolvedRuntimePolicy,
    runtime_implementation_fingerprint: String,
) -> ExecutionPlan {
    try_execution_plan_with_policy_and_runtime_fingerprint(
        policy,
        runtime_implementation_fingerprint,
    )
    .unwrap()
}

pub(crate) fn try_execution_plan_with_policy_and_runtime_fingerprint(
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
pub(crate) struct TestBuffer {
    pub(crate) descriptor: BufferDescriptor,
    pub(crate) marker: String,
    pub(crate) drop_trace: Weak<Mutex<Trace>>,
    pub(crate) backend_lifetime: Weak<()>,
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
pub(crate) struct TestStream {
    pub(crate) synchronize_count: u64,
    pub(crate) state: StreamState,
    pub(crate) drop_trace: Weak<Mutex<Trace>>,
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
pub(crate) struct TestRuntimeError(pub(crate) &'static str);

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvalidCommit {
    Descriptor,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostAllocationBehavior {
    ForgetThenError,
    DropThenError,
    Panic,
}

#[derive(Default)]
pub(crate) struct Trace {
    pub(crate) calls: Vec<String>,
    pub(crate) runtime_allocate_calls: u64,
    pub(crate) drift_on_allocate: bool,
    pub(crate) drift_on_create_stream: bool,
    pub(crate) drift_on_submit: bool,
    pub(crate) drift_on_synchronize: bool,
    pub(crate) synchronize_failures: u32,
    pub(crate) synchronize_returns_not_ready: bool,
    pub(crate) panic_on_stream_state: bool,
    pub(crate) synchronize_block: Option<(Arc<Barrier>, Arc<Barrier>)>,
    pub(crate) runtime_synchronize_calls: u64,
    pub(crate) failures: BTreeMap<String, u32>,
    pub(crate) invalid_commits: BTreeMap<String, InvalidCommit>,
    pub(crate) post_allocation: BTreeMap<String, PostAllocationBehavior>,
    pub(crate) abandon: Vec<ResourceAbandonSignal>,
    pub(crate) quarantine_sizes: Vec<usize>,
    pub(crate) quarantine_actual_mismatch: Vec<bool>,
    pub(crate) durable_ownership: Vec<ResourcePoolOwnership<TestRuntime>>,
    pub(crate) abandon_claimed_bytes: Vec<u64>,
    pub(crate) abandon_buffer_counts: Vec<usize>,
    pub(crate) retain_ownership: bool,
    pub(crate) panic_on_abandon: bool,
    pub(crate) buffer_drops: u64,
    pub(crate) buffer_drops_after_backend: u64,
    pub(crate) stream_drops: u64,
}

#[derive(Clone)]
pub(crate) struct TestRuntime {
    pub(crate) descriptor: DeviceDescriptor,
    pub(crate) alternate_descriptor: DeviceDescriptor,
    pub(crate) use_alternate_descriptor: Arc<AtomicBool>,
    pub(crate) trace: Arc<Mutex<Trace>>,
    pub(crate) backend_lifetime: Arc<()>,
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
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        assert!(!commands.is_empty(), "core must not submit an empty batch");
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
pub(crate) struct TestDriver {
    pub(crate) device_id: DeviceId,
    pub(crate) device_runtime_implementation_fingerprint: String,
    pub(crate) device_capacity_bytes: u64,
    pub(crate) trace: Arc<Mutex<Trace>>,
    pub(crate) runtime: Arc<TestRuntime>,
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

pub(crate) fn configured_driver(
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

pub(crate) fn sequence_runtime(plan: &ExecutionPlan) -> Arc<TestRuntime> {
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

pub(crate) fn transaction(
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

pub(crate) fn plan_runtime(
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

pub(crate) fn admit_logical_request(
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

pub(crate) fn admit_logical_sequence(
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

pub(crate) fn admit_logical_child_sequence(
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

pub(crate) fn close_plan_runtime(
    root: Arc<PlanRuntimeResources<TestRuntime>>,
) -> PlanRuntimeCloseReceipt {
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
}

pub(crate) fn admit_resources(
    plan: &ExecutionPlan,
    request_id: RequestIdentity,
) -> Result<StaticProvisioningPermit<TestRuntime>, VNextError> {
    required_static(plan, sequence_runtime(plan), request_id)
}

pub(crate) fn required_static(
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

pub(crate) fn plan_resources(plan: &ExecutionPlan) -> Vec<ResourceId> {
    plan.payload()
        .memory()
        .static_allocations()
        .iter()
        .map(|allocation| allocation.resource_id().clone())
        .collect()
}

pub(crate) fn failure_key(action: &str, resource: &ResourceId) -> String {
    format!("{action}:{}", resource.as_str())
}

pub(crate) fn calls(trace: &Arc<Mutex<Trace>>, prefix: &str) -> Vec<String> {
    trace
        .lock()
        .unwrap()
        .calls
        .iter()
        .filter(|call| call.starts_with(prefix))
        .cloned()
        .collect()
}

pub(crate) fn expect_err<T, E>(result: Result<T, E>) -> E {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

pub(crate) fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}

pub(crate) fn canonical_json(value: Value) -> Value {
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

pub(crate) fn rehash_plan_json(value: &mut Value) {
    let payload = value["payload"].as_object_mut().unwrap();
    payload.remove("plan_id");
    let material = canonical_json(Value::Object(payload.clone()));
    let bytes = serde_json::to_vec(&material).unwrap();
    let digest = format!("{:x}", Sha256::digest(bytes));
    value["payload"]["plan_id"] = json!(format!("plan/sha256/{digest}"));
    value["plan_hash"] = json!(digest);
}
