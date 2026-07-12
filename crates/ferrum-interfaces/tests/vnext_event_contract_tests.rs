use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const MAX_EVENT_MAINTENANCE_ATTEMPTS: usize = 8;

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
    #[serde(default)]
    no_static: bool,
}

#[derive(Default)]
struct TestFamily;

impl ModelFamilyProvider for TestFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.event-contract"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.event")])
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
        Ok(id("metadata.event"))
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
                reason: "event fixture requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        let required = !config.no_static;
        Ok(WeightSchema {
            format_id: id("weight-format.event-dense"),
            layout_id: id("weight-layout.event-dense"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component"),
                role: WeightComponentRole::Values,
                external_names: vec!["weight.bin".to_owned()],
                dimensions: vec![config.width],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![config.width],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component"),
                },
                required,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let mut inputs = vec![id("value.input")];
        if config.no_static {
            inputs.push(id("value.weight"));
        }
        let weights = if config.no_static {
            Vec::new()
        } else {
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }]
        };
        ModelProgram::new(
            self.family_id().clone(),
            inputs,
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.first"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        inputs: vec![id("value.input"), id("value.weight")],
                        outputs: vec![id("value.middle")],
                        attributes: BTreeMap::new(),
                    },
                    ProgramNode {
                        id: id("node.second"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        inputs: vec![id("value.middle"), id("value.weight")],
                        outputs: vec![id("value.output")],
                        attributes: BTreeMap::new(),
                    },
                ],
            }],
            Vec::new(),
            weights,
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

fn tensor_contract(access: TensorAccess) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(4)],
        BTreeSet::from([ElementType::F32]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

fn operation() -> OperationDescriptor {
    OperationDescriptor {
        id: id("operation.main"),
        version: ContractVersion::new(1, 0),
        inputs: vec![
            tensor_contract(TensorAccess::Read),
            tensor_contract(TensorAccess::Read),
        ],
        outputs: vec![tensor_contract(TensorAccess::Write)],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
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
    let operation = operation();
    let device_id: DeviceId = id("device.event.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.event"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        ContractVersion::new(1, 0),
        device_id.clone(),
        capabilities.clone(),
        BTreeSet::from([id("weight-format.event-dense")]),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation),
        "resource-estimator.event",
        ContractVersion::new(1, 0),
        sha('b'),
    )
    .unwrap();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: sha('d'),
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        vec![operation.clone()],
        BTreeMap::from([(operation.id.clone(), vec![provider])]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.event"),
            ContractVersion::new(1, 0),
            sha('d'),
            device_id,
            capabilities,
        )
        .unwrap()],
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
                reason: "event fixture operation signature mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

fn policy() -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.event",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 4096,
            reserve_bytes: 128,
            maximum_active_sequences: 2,
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

fn resolved_tensor() -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
}

fn binding(
    value: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    usage: BufferUsage,
    resource: &str,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id(value),
        role,
        ordinal,
        resolved_tensor(),
        if role == ResolvedValueRole::Output {
            TensorAccess::Write
        } else {
            TensorAccess::Read
        },
        AliasPolicy::NoAlias,
        usage,
        ResolvedValueStorage::single(id(resource), 0, 16, ElementType::F32).unwrap(),
    )
    .unwrap()
}

fn make_operation_registry(catalog: &CapabilityCatalog) -> OperationRuntimeRegistry<TestRuntime> {
    OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
        })],
        vec![Box::new(TestExecutionProvider::new(catalog))],
    )
    .unwrap()
}

fn execution_plan(
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    execution_plan_with_mode(suffix, operation_registry, false)
}

fn no_static_execution_plan(
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    execution_plan_with_mode(suffix, operation_registry, true)
}

fn execution_plan_with_mode(
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
    no_static: bool,
) -> ExecutionPlan {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4, "no_static": no_static}))
        .unwrap();
    let catalog = catalog();
    let policy = policy();
    let weight = || {
        if no_static {
            binding(
                "value.weight",
                ResolvedValueRole::Input,
                1,
                BufferUsage::Activations,
                &format!("resource.weight.{suffix}"),
            )
        } else {
            ResolvedValueBinding::new(
                id("value.weight"),
                ResolvedValueRole::Input,
                1,
                resolved_tensor(),
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Weights,
                ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                    Some(id("weight.component")),
                    id(format!("resource.weight.{suffix}")),
                    0,
                    16,
                    ElementType::F32,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap()
        }
    };
    let first_values = vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.input.{suffix}"),
        ),
        weight(),
        binding(
            "value.middle",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
    ];
    let second_values = vec![
        binding(
            "value.middle",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
        weight(),
        binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.output.{suffix}"),
        ),
    ];
    let planning = operation_registry.planning();
    let first = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        id("node.first"),
        first_values,
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let second = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        id("node.second"),
        second_values,
        BTreeSet::new(),
        None,
    )
    .unwrap();
    ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![first, second]).unwrap(),
    )
    .unwrap()
}

#[derive(Debug)]
struct TestBuffer {
    descriptor: BufferDescriptor,
}

#[derive(Debug)]
struct TestStream {
    synchronizations: u64,
    failed: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestFence {
    Succeeded,
    Failed,
    Indeterminate,
    ContractFailed,
}

#[derive(Debug)]
struct TestRuntimeError;

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("test runtime failure")
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone)]
struct TestRuntime {
    descriptor: DeviceDescriptor,
    fail_next_fence: Arc<AtomicBool>,
    indeterminate_next_fence: Arc<AtomicBool>,
    contract_fail_next_fence: Arc<AtomicBool>,
    panic_next_submit: Arc<AtomicBool>,
    synchronize_fails: Arc<AtomicBool>,
    stream_failed: Arc<AtomicBool>,
}

impl TestRuntime {
    fn fail_next_fence(&self) {
        assert!(!self.fail_next_fence.swap(true, Ordering::SeqCst));
    }

    fn make_next_fence_indeterminate(&self) {
        assert!(!self.indeterminate_next_fence.swap(true, Ordering::SeqCst));
    }

    fn contract_fail_next_fence(&self) {
        assert!(!self.contract_fail_next_fence.swap(true, Ordering::SeqCst));
    }

    fn panic_next_submit(&self) {
        assert!(!self.panic_next_submit.swap(true, Ordering::SeqCst));
    }

    fn set_synchronize_fails(&self, fails: bool) {
        self.synchronize_fails.store(fails, Ordering::SeqCst);
    }

    fn reset_stream_failure(&self) {
        self.stream_failed.store(false, Ordering::SeqCst);
    }
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = ();
    type Fence = TestFence;
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        Ok(TestBuffer {
            descriptor: BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(TestStream {
            synchronizations: 0,
            failed: Arc::clone(&self.stream_failed),
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.failed.load(Ordering::SeqCst) {
            StreamState::Failed
        } else {
            StreamState::Ready
        }
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
        _command: Self::Command,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        assert!(
            !self.panic_next_submit.swap(false, Ordering::SeqCst),
            "injected event submit panic"
        );
        Ok(
            if self.indeterminate_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::Indeterminate
            } else if self.contract_fail_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::ContractFailed
            } else if self.fail_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::Failed
            } else {
                TestFence::Succeeded
            },
        )
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        match fence {
            TestFence::Succeeded => FenceQuery::Terminal(DeviceTerminal::Succeeded),
            TestFence::Failed => {
                FenceQuery::Terminal(DeviceTerminal::FailedButQuiescent(TestRuntimeError))
            }
            TestFence::Indeterminate => FenceQuery::Indeterminate(TestRuntimeError),
            TestFence::ContractFailed => {
                self.stream_failed.store(true, Ordering::SeqCst);
                FenceQuery::Terminal(DeviceTerminal::Succeeded)
            }
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminal<Self::Error>, FenceIndeterminate<Self::Error>> {
        match fence {
            TestFence::Succeeded => Ok(DeviceTerminal::Succeeded),
            TestFence::Failed => Ok(DeviceTerminal::FailedButQuiescent(TestRuntimeError)),
            TestFence::Indeterminate => Err(FenceIndeterminate::new(TestRuntimeError)),
            TestFence::ContractFailed => {
                self.stream_failed.store(true, Ordering::SeqCst);
                Ok(DeviceTerminal::Succeeded)
            }
        }
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        stream.synchronizations += 1;
        if self.synchronize_fails.load(Ordering::SeqCst) {
            return Err(TestRuntimeError);
        }
        stream.failed.store(false, Ordering::SeqCst);
        Ok(())
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

struct TestExecutionProvider {
    descriptor: OperationProviderDescriptor,
}

impl TestExecutionProvider {
    fn new(catalog: &CapabilityCatalog) -> Self {
        Self {
            descriptor: catalog.providers_for(&id("operation.main")).unwrap()[0].clone(),
        }
    }
}

impl OperationResourceEstimator for TestExecutionProvider {
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
            None,
            None,
        ))
    }
}

impl OperationProvider<TestRuntime> for TestExecutionProvider {
    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_and_submit_single(
    provider: &BoundOperationProvider<'_, TestRuntime>,
    resolved: &ResolvedModelPlan,
    identity: &ExecutionIdentityEnvelope,
    frame_id: &ExecutionFrameId,
    node_invocation_id: &NodeInvocationId,
    node_id: &NodeId,
    active: &TrustedActiveSequenceBinding,
    invocation: InvocationResourceLease<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
    let parts = identity.parts();
    if parts.frame_id != Some(*frame_id)
        || parts.node_invocation_id != Some(*node_invocation_id)
        || parts.node_id.as_ref() != Some(node_id)
    {
        return Err(OperationDispatchError::Contract(
            VNextError::InvalidExecutionPlan {
                reason: "single-participant dispatch arguments disagree".to_owned(),
            },
        ));
    }
    let active_bindings = std::slice::from_ref(active);
    let batch_identity = OperationDispatch::bind_batch_identity(
        resolved,
        vec![identity.clone()],
        active_bindings,
        &invocation,
        lane,
    )
    .map_err(OperationDispatchError::Contract)?;
    OperationDispatch::encode_and_submit(
        provider,
        resolved,
        &batch_identity,
        active_bindings,
        invocation,
        lane,
        reaper,
    )
}

#[derive(Debug, Default)]
struct DriverTrace {
    reconciles: usize,
    quarantines: usize,
    abandon: Vec<ResourceAbandonSignal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitBehavior {
    Valid,
    InvalidFirst,
}

#[derive(Debug)]
struct TestDriver {
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    device_capacity_bytes: u64,
    behavior: CommitBehavior,
    invalid_returned: bool,
    trace: Arc<Mutex<DriverTrace>>,
    runtime: Arc<TestRuntime>,
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
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        if self.behavior == CommitBehavior::InvalidFirst && !self.invalid_returned {
            self.invalid_returned = true;
            return Err(ResourceDriverFailure::new(
                FailureEnvelope::new(
                    FailureDomain::Resource,
                    "commit_injected_failure",
                    "injected commit failure",
                    true,
                )
                .unwrap(),
            )
            .unwrap());
        }
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        context.allocate(&request).map_err(|error| {
            ResourceDriverFailure::new(
                FailureEnvelope::new(
                    FailureDomain::Resource,
                    "commit_allocation_failed",
                    format!("{error:?}"),
                    false,
                )
                .unwrap(),
            )
            .unwrap()
        })
    }

    fn compensate_reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn compensate_commit_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.descriptor.resource_id, *reservation.resource_id());
        Ok(())
    }

    fn rollback_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn release_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.descriptor.resource_id, *reservation.resource_id());
        Ok(())
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _expected: &ResourceReservation,
        _actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        self.trace.lock().unwrap().reconciles += 1;
        Ok(())
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        assert!(!ownership.buffers().is_empty());
        self.trace.lock().unwrap().quarantines += 1;
        drop(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        self.trace.lock().unwrap().abandon.push(
            ownership
                .abandon_signal()
                .expect("abandon must carry signal")
                .clone(),
        );
        drop(ownership);
    }
}

fn driver(plan: &ExecutionPlan, behavior: CommitBehavior) -> (TestDriver, Arc<Mutex<DriverTrace>>) {
    let trace = Arc::new(Mutex::new(DriverTrace::default()));
    let runtime = Arc::new(TestRuntime {
        descriptor: DeviceDescriptor {
            id: plan.payload().device_id().clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: plan.payload().memory().device_capacity_bytes(),
            runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            capabilities: BTreeSet::from([id("capability.compute")]),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        fail_next_fence: Arc::new(AtomicBool::new(false)),
        indeterminate_next_fence: Arc::new(AtomicBool::new(false)),
        contract_fail_next_fence: Arc::new(AtomicBool::new(false)),
        panic_next_submit: Arc::new(AtomicBool::new(false)),
        synchronize_fails: Arc::new(AtomicBool::new(false)),
        stream_failed: Arc::new(AtomicBool::new(false)),
    });
    (
        TestDriver {
            device_id: plan.payload().device_id().clone(),
            device_runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            device_capacity_bytes: plan.payload().memory().device_capacity_bytes(),
            behavior,
            invalid_returned: false,
            trace: trace.clone(),
            runtime,
        },
        trace,
    )
}

fn transaction(
    plan: &ExecutionPlan,
    run_id: &str,
    transaction_id: &str,
    provisioning_request: &str,
    behavior: CommitBehavior,
) -> (
    ResourceTransaction<TestDriver, TransactionNew>,
    Arc<Mutex<DriverTrace>>,
    Arc<TestRuntime>,
) {
    let request: RequestIdentity = id(provisioning_request);
    let (driver, trace) = driver(plan, behavior);
    let runtime = Arc::clone(driver.runtime());
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(Arc::clone(driver.runtime()), request.clone())
        .unwrap()
        .into_parts();
    let permit = match provisioning {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => {
            panic!("event fixture unexpectedly has no physical provisioning")
        }
    };
    let pool_ids = permit
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    permit
        .maintenance_controller()
        .initialize_pools(&pool_ids)
        .unwrap();
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id(run_id),
        id(transaction_id),
    );
    (
        ResourceTransaction::begin(driver, identity, permit).unwrap(),
        trace,
        runtime,
    )
}

#[derive(Default)]
struct RecordingSink {
    kinds: Mutex<Vec<ExecutionEventKind>>,
}

impl ExecutionEventSink for RecordingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(
        &self,
        event: &ExecutionEvent,
        _permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        self.kinds.lock().unwrap().push(event.kind());
        Ok(())
    }
}

struct FailingSink;

impl ExecutionEventSink for FailingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(
        &self,
        _event: &ExecutionEvent,
        _permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        Err(ExecutionEventSinkError::new("injected sink failure"))
    }
}

fn base_parts(
    run_id: &RunId,
    request_id: &RequestIdentity,
    sequence: u64,
    span_id: impl Into<String>,
    parent_span_id: Option<SpanId>,
) -> ExecutionIdentityParts {
    ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: run_id.clone(),
        request_id: request_id.clone(),
        sequence,
        plan_id: None,
        plan_hash: None,
        frame_id: None,
        node_invocation_id: None,
        node_id: None,
        operation_id: None,
        provider_id: None,
        device_id: None,
        resource_pool_id: None,
        resource_pool_identity_fingerprint: None,
        provisioning_run_id: None,
        provisioning_request_id: None,
        transaction_id: None,
        active_sequence_slot: None,
        admission_generation: None,
        activation_epoch: None,
        runtime_implementation_fingerprint: None,
        active_sequence_fingerprint: None,
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id(span_id),
        parent_span_id,
        async_links: Vec::new(),
    }
}

fn bind_plan(mut parts: ExecutionIdentityParts, plan: &ExecutionPlan) -> ExecutionIdentityParts {
    parts.plan_id = Some(plan.payload().plan_id().clone());
    parts.plan_hash = Some(plan.plan_hash().clone());
    parts.device_id = Some(plan.payload().device_id().clone());
    parts.runtime_implementation_fingerprint = Some(
        plan.payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
    );
    parts
}

fn bind_active(
    mut parts: ExecutionIdentityParts,
    active: &TrustedActiveSequenceBinding,
) -> ExecutionIdentityParts {
    let provisioning = active.static_provisioning_identity();
    parts.resource_pool_id = active.static_pool_id();
    parts.resource_pool_identity_fingerprint = active.static_pool_identity_fingerprint();
    parts.provisioning_run_id = provisioning.map(|identity| identity.run_id().clone());
    parts.provisioning_request_id = provisioning.map(|identity| identity.request_id().clone());
    parts.transaction_id = provisioning.map(|identity| identity.transaction_id().clone());
    parts.active_sequence_slot = Some(active.sequence_authority().sparse_id());
    parts.admission_generation = Some(active.sequence_authority().generation());
    parts.activation_epoch = Some(active.activation_epoch());
    debug_assert_eq!(
        parts.runtime_implementation_fingerprint.as_deref(),
        Some(active.runtime_implementation_fingerprint())
    );
    parts.active_sequence_fingerprint = Some(active.fingerprint().to_owned());
    parts
}

fn make_event(
    sequence: u64,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    parts: ExecutionIdentityParts,
    detail: ExecutionEventDetail,
) -> ExecutionEvent {
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        phase,
        kind,
        ExecutionIdentityEnvelope::new(parts).unwrap(),
        detail,
    )
    .unwrap()
}

fn accepted_event(run: &RunId, request: &RequestIdentity) -> ExecutionEvent {
    make_event(
        1,
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestAccepted,
        base_parts(run, request, 1, "span.request", None),
        ExecutionEventDetail::None,
    )
}

fn plan_event(plan: &ExecutionPlan, run: &RunId, request: &RequestIdentity) -> ExecutionEvent {
    make_event(
        2,
        ExecutionPhase::Planning,
        ExecutionEventKind::PlanBuilt,
        bind_plan(
            base_parts(run, request, 2, "span.plan", Some(id("span.request"))),
            plan,
        ),
        ExecutionEventDetail::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn frame_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
    frame: u64,
    kind: ExecutionEventKind,
) -> ExecutionEvent {
    let frame_id = ExecutionFrameId::try_from(frame).unwrap();
    let span: SpanId = id(format!("span.frame.{frame}"));
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                run,
                request,
                sequence,
                span.as_str(),
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.frame_id = Some(frame_id);
    make_event(
        sequence,
        ExecutionPhase::Execution,
        kind,
        parts,
        ExecutionEventDetail::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn node_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
    frame: u64,
    invocation: u64,
    node_index: usize,
    kind: ExecutionEventKind,
) -> ExecutionEvent {
    let node = &plan.payload().nodes()[node_index];
    let frame_span: SpanId = id(format!("span.frame.{frame}"));
    let node_span: SpanId = id(format!("span.frame.{frame}.node.{node_index}"));
    let operation_span: SpanId = id(format!("span.frame.{frame}.operation.{node_index}"));
    let (span, parent) = match kind {
        ExecutionEventKind::NodeStarted | ExecutionEventKind::NodeRetired => {
            (node_span, frame_span)
        }
        ExecutionEventKind::OperationSubmitted => (operation_span, node_span),
        _ => panic!("invalid node event kind"),
    };
    let mut parts = bind_active(
        bind_plan(
            base_parts(run, request, sequence, span.as_str(), Some(parent)),
            plan,
        ),
        active,
    );
    parts.frame_id = Some(ExecutionFrameId::try_from(frame).unwrap());
    parts.node_invocation_id = Some(NodeInvocationId::try_from(invocation).unwrap());
    parts.node_id = Some(node.id().clone());
    parts.operation_id = Some(node.operation_id().clone());
    parts.provider_id = Some(node.selection().selected_provider().clone());
    make_event(
        sequence,
        ExecutionPhase::Execution,
        kind,
        parts,
        ExecutionEventDetail::None,
    )
}

fn request_completed_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(run, request, sequence, "span.request", None),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::RequestCompleted,
        parts,
        ExecutionEventDetail::Counters {
            input: 11,
            output: 7,
        },
    )
}

fn sequence_completed_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.sequence-completed",
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::SequenceCompleted,
        parts,
        ExecutionEventDetail::None,
    )
}

fn sequence_aborted_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    aborted: &TrustedAbortedSequenceBinding,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.sequence-aborted",
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.aborted_sequence_fingerprint = Some(aborted.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::SequenceAborted,
        parts,
        ExecutionEventDetail::None,
    )
}

fn operation_failure_event(failure: &IdentifiedFailure, sequence: u64) -> ExecutionEvent {
    let failed_operation = failure.identity().parts();
    let mut observation = failed_operation.clone();
    observation.sequence = sequence;
    observation.span_id = id(format!("{}.failure-observed", failed_operation.span_id));
    observation.parent_span_id = Some(failed_operation.span_id.clone());
    let identity = ExecutionIdentityEnvelope::new(observation).unwrap();
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        ExecutionPhase::Execution,
        ExecutionEventKind::FailureObserved,
        identity,
        ExecutionEventDetail::Failure(failure.clone()),
    )
    .unwrap()
}

fn planning_failure_event(
    plan: &ExecutionPlan,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
) -> (ExecutionEvent, IdentifiedFailure) {
    let identity = ExecutionIdentityEnvelope::new(bind_plan(
        base_parts(
            run,
            request,
            sequence,
            "span.planning-failure",
            Some(id("span.request")),
        ),
        plan,
    ))
    .unwrap();
    let failure = IdentifiedFailure::new(
        identity.clone(),
        FailureEnvelope::new(
            FailureDomain::Planning,
            "planning_fixture_failure",
            "injected planning failure",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let event = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        ExecutionPhase::Planning,
        ExecutionEventKind::FailureObserved,
        identity,
        ExecutionEventDetail::Failure(failure.clone()),
    )
    .unwrap();
    (event, failure)
}

fn request_failed_terminal_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: Option<&TrustedCompletedSequenceBinding>,
    aborted: Option<&TrustedAbortedSequenceBinding>,
    failure: &IdentifiedFailure,
    sequence: u64,
) -> ExecutionEvent {
    assert!(completed.is_some() ^ aborted.is_some());
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.request",
                None,
            ),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint =
        completed.map(|binding| binding.fingerprint().to_owned());
    parts.aborted_sequence_fingerprint = aborted.map(|binding| binding.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::RequestFailed,
        parts,
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: failure.fingerprint(),
        },
    )
}

fn request_journal(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    submissions: &[SubmittedOperationReceipt],
    completions: &[OperationCompletionReceipt],
    frames: u64,
) -> Vec<ExecutionEvent> {
    let run = active.run_id();
    let request = active.request_id();
    let mut events = vec![accepted_event(run, request), plan_event(plan, run, request)];
    let mut sequence = 3_u64;
    let mut invocation = 1_u64;
    let mut submission_index = 0_usize;
    let mut completion_index = 0_usize;
    for frame in 1..=frames {
        events.push(frame_event(
            plan,
            active,
            run,
            request,
            sequence,
            frame,
            ExecutionEventKind::FrameStarted,
        ));
        sequence += 1;
        for node_index in 0..plan.payload().nodes().len() {
            for kind in [
                ExecutionEventKind::NodeStarted,
                ExecutionEventKind::OperationSubmitted,
                ExecutionEventKind::NodeRetired,
            ] {
                let event = node_event(
                    plan, active, run, request, sequence, frame, invocation, node_index, kind,
                );
                if kind == ExecutionEventKind::OperationSubmitted {
                    assert_eq!(
                        submissions[submission_index].participants()[0].identity(),
                        event.identity()
                    );
                    submission_index += 1;
                } else if kind == ExecutionEventKind::NodeRetired {
                    assert!(same_operation_authority(
                        event.identity(),
                        completions[completion_index].participants()[0]
                            .submission()
                            .identity(),
                    ));
                    completion_index += 1;
                }
                events.push(event);
                sequence += 1;
            }
            invocation += 1;
        }
        events.push(frame_event(
            plan,
            active,
            run,
            request,
            sequence,
            frame,
            ExecutionEventKind::FrameCompleted,
        ));
        sequence += 1;
    }
    assert_eq!(submission_index, submissions.len());
    assert_eq!(completion_index, completions.len());
    events.push(sequence_completed_event(plan, active, completed, sequence));
    sequence += 1;
    events.push(request_completed_event(
        plan, active, completed, run, request, sequence,
    ));
    events
}

fn same_operation_authority(
    observation: &ExecutionIdentityEnvelope,
    operation: &ExecutionIdentityEnvelope,
) -> bool {
    let mut normalized = observation.parts().clone();
    normalized.sequence = operation.parts().sequence;
    normalized.span_id = operation.parts().span_id.clone();
    normalized.parent_span_id = operation.parts().parent_span_id.clone();
    normalized == *operation.parts()
}

fn event_context<'a>(
    event: &'a ExecutionEvent,
    topology: &'a TrustedExecutionTopology,
    active: &'a TrustedActiveSequenceBinding,
    completed: &'a TrustedCompletedSequenceBinding,
    submissions: &'a [SubmittedOperationReceipt],
    completions: &'a [OperationCompletionReceipt],
) -> TrustedExecutionEventContext<'a> {
    match event.kind() {
        ExecutionEventKind::RequestAccepted => {
            TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id())
        }
        ExecutionEventKind::PlanBuilt => {
            TrustedExecutionEventContext::bound(active.run_id(), active.request_id(), topology)
        }
        ExecutionEventKind::RequestFailed => {
            let failure = match event.detail() {
                ExecutionEventDetail::Failure(failure) => failure,
                _ => unreachable!(),
            };
            TrustedExecutionEventContext::failure(
                active.run_id(),
                active.request_id(),
                Some(topology),
                event
                    .identity()
                    .parts()
                    .active_sequence_slot
                    .is_some()
                    .then_some(active),
                failure,
            )
        }
        ExecutionEventKind::OperationSubmitted => {
            let receipt = submissions
                .iter()
                .find(|receipt| {
                    receipt
                        .participants()
                        .iter()
                        .any(|participant| participant.identity() == event.identity())
                })
                .expect("journal operation has external submission receipt");
            TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                receipt,
            )
        }
        ExecutionEventKind::NodeRetired => {
            let completion = completions
                .iter()
                .flat_map(|receipt| receipt.participants())
                .find(|participant| {
                    same_operation_authority(event.identity(), participant.submission().identity())
                })
                .expect("retired node has external participant completion evidence");
            TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                completion,
            )
        }
        ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
            TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                completed,
            )
        }
        _ => TrustedExecutionEventContext::active(
            active.run_id(),
            active.request_id(),
            topology,
            active,
        ),
    }
}

fn observe_journal(
    journal: &[ExecutionEvent],
    topology: &TrustedExecutionTopology,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    submissions: &[SubmittedOperationReceipt],
    completions: &[OperationCompletionReceipt],
) -> Result<ExecutionEventCursor, VNextError> {
    let mut cursor =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in journal {
        cursor.observe_against(
            event,
            &event_context(event, topology, active, completed, submissions, completions),
        )?;
    }
    Ok(cursor)
}

fn observe_failure_journal(
    evidence: &FailureSequenceEvidence,
    topology: &TrustedExecutionTopology,
) -> Result<ExecutionEventCursor, VNextError> {
    let mut cursor = ExecutionEventCursor::new(
        evidence.active.run_id().clone(),
        evidence.active.request_id().clone(),
    );
    let mut first_failure: Option<IdentifiedFailure> = None;
    for event in &evidence.journal {
        let context = match event.kind() {
            ExecutionEventKind::RequestAccepted => TrustedExecutionEventContext::pre_plan(
                evidence.active.run_id(),
                evidence.active.request_id(),
            ),
            ExecutionEventKind::PlanBuilt => TrustedExecutionEventContext::bound(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
            ),
            ExecutionEventKind::OperationSubmitted => {
                let submission = evidence
                    .submissions
                    .iter()
                    .find(|receipt| {
                        receipt
                            .participants()
                            .iter()
                            .any(|participant| participant.identity() == event.identity())
                    })
                    .unwrap();
                TrustedExecutionEventContext::operation_submitted(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    submission,
                )
            }
            ExecutionEventKind::NodeRetired => {
                let completion = evidence
                    .completions
                    .iter()
                    .flat_map(|receipt| receipt.participants())
                    .find(|participant| {
                        same_operation_authority(
                            event.identity(),
                            participant.submission().identity(),
                        )
                    })
                    .unwrap();
                TrustedExecutionEventContext::node_retired(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    completion,
                )
            }
            ExecutionEventKind::FailureObserved => {
                let failure = match event.detail() {
                    ExecutionEventDetail::Failure(failure) => failure,
                    _ => unreachable!(),
                };
                TrustedExecutionEventContext::failure(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    Some(topology),
                    Some(&evidence.active),
                    failure,
                )
            }
            ExecutionEventKind::SequenceCompleted => TrustedExecutionEventContext::completed(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
                evidence.completed.as_ref().unwrap(),
            ),
            ExecutionEventKind::SequenceAborted => TrustedExecutionEventContext::aborted(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
                evidence.aborted.as_ref().unwrap(),
            ),
            ExecutionEventKind::RequestFailed => {
                TrustedExecutionEventContext::failure_with_disposition(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    evidence.completed.as_ref(),
                    evidence.aborted.as_ref(),
                    first_failure.as_ref().unwrap(),
                )
            }
            _ => TrustedExecutionEventContext::active(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
            ),
        };
        cursor.observe_against(event, &context)?;
        if event.kind() == ExecutionEventKind::FailureObserved {
            let ExecutionEventDetail::Failure(failure) = event.detail() else {
                unreachable!()
            };
            first_failure = Some(failure.clone());
        }
    }
    Ok(cursor)
}

struct EventModelRegistry {
    registration: TypedFamilyRegistration<TestFamily>,
}

impl EventModelRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }
}

impl ModelFamilyRegistry for EventModelRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

fn plan_resolutions_with_mode(
    suffix: &str,
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime: &ResolvedRuntimePolicy,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
    no_static: bool,
) -> Vec<PlanNodeResolution> {
    let weight = || {
        if no_static {
            binding(
                "value.weight",
                ResolvedValueRole::Input,
                1,
                BufferUsage::Activations,
                &format!("resource.weight.{suffix}"),
            )
        } else {
            ResolvedValueBinding::new(
                id("value.weight"),
                ResolvedValueRole::Input,
                1,
                resolved_tensor(),
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Weights,
                ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                    Some(id("weight.component")),
                    id(format!("resource.weight.{suffix}")),
                    0,
                    16,
                    ElementType::F32,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap()
        }
    };
    let first_values = vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.input.{suffix}"),
        ),
        weight(),
        binding(
            "value.middle",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
    ];
    let second_values = vec![
        binding(
            "value.middle",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
        weight(),
        binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.output.{suffix}"),
        ),
    ];
    let planning = operation_registry.planning();
    [("node.first", first_values), ("node.second", second_values)]
        .into_iter()
        .map(|(node, values)| {
            PlanNodeResolution::resolve(
                family,
                catalog,
                runtime,
                &planning,
                id(node),
                values,
                BTreeSet::new(),
                None,
            )
            .unwrap()
        })
        .collect()
}

const RESOLUTION_FIELDS: [ResolutionField; 20] = [
    ResolutionField::OriginalSource,
    ResolutionField::ResolvedSource,
    ResolutionField::Config,
    ResolutionField::ExternalMetadata,
    ResolutionField::Family,
    ResolutionField::WeightSchema,
    ResolutionField::WeightFormat,
    ResolutionField::Tokenizer,
    ResolutionField::Template,
    ResolutionField::SpecialTokens,
    ResolutionField::Device,
    ResolutionField::Capabilities,
    ResolutionField::RuntimePreset,
    ResolutionField::RuntimeMemory,
    ResolutionField::Admission,
    ResolutionField::Engine,
    ResolutionField::ExecutionPlan,
    ResolutionField::Sampling,
    ResolutionField::Stop,
    ResolutionField::StructuredOutput,
];

fn resolution_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSource => ResolutionDecisionSource::UserInput,
        ResolutionField::ResolvedSource
        | ResolutionField::Config
        | ResolutionField::ExternalMetadata
        | ResolutionField::Family
        | ResolutionField::WeightSchema
        | ResolutionField::WeightFormat
        | ResolutionField::Tokenizer
        | ResolutionField::Template
        | ResolutionField::SpecialTokens => ResolutionDecisionSource::TypedModelResolution,
        ResolutionField::Device | ResolutionField::Capabilities | ResolutionField::Engine => {
            ResolutionDecisionSource::CapabilityResolution
        }
        ResolutionField::RuntimePreset
        | ResolutionField::RuntimeMemory
        | ResolutionField::Admission => ResolutionDecisionSource::RuntimePreset,
        ResolutionField::ExecutionPlan => ResolutionDecisionSource::Planner,
        ResolutionField::Sampling | ResolutionField::Stop | ResolutionField::StructuredOutput => {
            ResolutionDecisionSource::ProductDefault
        }
    }
}

fn resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
    match field {
        ResolutionField::OriginalSource => serde_json::to_value(&inputs.original_source).unwrap(),
        ResolutionField::ResolvedSource => serde_json::to_value(&inputs.resolved_source).unwrap(),
        ResolutionField::Config => serde_json::to_value(&inputs.config).unwrap(),
        ResolutionField::ExternalMetadata => {
            serde_json::to_value(&inputs.external_metadata_id).unwrap()
        }
        ResolutionField::Family => {
            serde_json::to_value(inputs.prepared_family.family_id()).unwrap()
        }
        ResolutionField::WeightSchema => {
            serde_json::to_value(inputs.prepared_family.weight_schema()).unwrap()
        }
        ResolutionField::WeightFormat => {
            serde_json::to_value(&inputs.prepared_family.weight_schema().format_id).unwrap()
        }
        ResolutionField::Tokenizer => serde_json::to_value(&inputs.tokenizer).unwrap(),
        ResolutionField::Template => {
            serde_json::to_value(&inputs.prepared_family.metadata().template).unwrap()
        }
        ResolutionField::SpecialTokens => {
            serde_json::to_value(&inputs.prepared_family.metadata().special_tokens).unwrap()
        }
        ResolutionField::Device => serde_json::to_value(&inputs.device).unwrap(),
        ResolutionField::Capabilities => serde_json::to_value(&inputs.capabilities).unwrap(),
        ResolutionField::RuntimePreset => json!({
            "policy_id": inputs.runtime.policy_id(),
            "version": inputs.runtime.version(),
            "scheduling": inputs.runtime.scheduling(),
        }),
        ResolutionField::RuntimeMemory => serde_json::to_value(inputs.runtime.memory()).unwrap(),
        ResolutionField::Admission => serde_json::to_value(inputs.runtime.admission()).unwrap(),
        ResolutionField::Engine => serde_json::to_value(&inputs.engine).unwrap(),
        ResolutionField::ExecutionPlan => json!(inputs.execution_plan.plan_hash().as_str()),
        ResolutionField::Sampling => serde_json::to_value(&inputs.sampling).unwrap(),
        ResolutionField::Stop => serde_json::to_value(&inputs.stop).unwrap(),
        ResolutionField::StructuredOutput => {
            serde_json::to_value(&inputs.structured_output).unwrap()
        }
    }
}

fn resolved_model_plan(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ResolvedModelPlan {
    resolved_model_plan_with_mode(plan, suffix, operation_registry, false)
}

fn no_static_resolved_model_plan(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ResolvedModelPlan {
    resolved_model_plan_with_mode(plan, suffix, operation_registry, true)
}

fn resolved_model_plan_with_mode(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
    no_static: bool,
) -> ResolvedModelPlan {
    let registry = EventModelRegistry::new();
    let family = registry
        .registration
        .prepare(&json!({"width": 4, "no_static": no_static}))
        .unwrap();
    let catalog = catalog();
    let runtime = policy();
    let resolutions = plan_resolutions_with_mode(
        suffix,
        &family,
        &catalog,
        &runtime,
        operation_registry,
        no_static,
    );
    let rebuilt = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &runtime, resolutions.clone()).unwrap(),
    )
    .unwrap();
    assert_eq!(&rebuilt, plan);

    let config_fingerprint = family.config_fingerprint().to_owned();
    let inputs = ResolvedModelPlanInputs {
        original_source: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/event-model".to_owned(),
            requested_revision: Some("main".to_owned()),
        },
        resolved_source: ResolvedModelSource {
            canonical_location: "repo/event-model".to_owned(),
            resolved_revision: "0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "config.json".to_owned(),
                    size_bytes: 11,
                    sha256: config_fingerprint.clone(),
                },
                FileFingerprint {
                    relative_path: "template.json".to_owned(),
                    size_bytes: 30,
                    sha256: sha('a'),
                },
                FileFingerprint {
                    relative_path: "tokenizer.json".to_owned(),
                    size_bytes: 20,
                    sha256: sha('b'),
                },
            ],
        },
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_fingerprint.clone(),
            typed_config_sha256: config_fingerprint,
        },
        external_metadata_id: id("metadata.event"),
        prepared_family: family.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.event"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1024,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.event"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('d'),
        },
        execution_plan: plan.clone(),
        sampling: SamplingPolicy::new(
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            None,
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            9271,
            TriStatePolicy::ModelDefault,
        )
        .unwrap(),
        stop: StopPolicy {
            maximum_output_tokens: 32,
            token_ids: BTreeSet::from([3]),
            strings: vec!["stop".to_owned()],
        },
        structured_output: StructuredOutputPolicy::JsonObject,
    };

    let mut bindings = Vec::new();
    let mut evidence = Vec::new();
    for (index, field) in RESOLUTION_FIELDS.into_iter().enumerate() {
        let source = resolution_source(field);
        let artifact_id: ResolutionArtifactId = id(format!("artifact.event.{index}"));
        let path = "/chosen".to_owned();
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "fixture.event".to_owned(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e'))
                        .unwrap(),
                    revision: "fixture-v1".to_owned(),
                    artifact_locator: format!("event/{index}"),
                },
                serde_json::to_vec(&json!({"chosen": resolution_value(&inputs, field)})).unwrap(),
                BTreeSet::from([path.clone()]),
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap(),
        );
        bindings.push(
            ResolutionDecisionBinding::new(
                field,
                source,
                id(format!("reason.event.{index}")),
                artifact_id,
                path,
            )
            .unwrap(),
        );
    }
    let context = ResolvedPlanValidationContext::new(
        &registry,
        &evidence,
        &resolutions,
        catalog.device(),
        &catalog,
        &runtime,
    );
    ResolvedModelPlan::new(inputs, bindings, &context).unwrap()
}

fn pool_timestamp(sequence: u64) -> MonotonicTimestamp {
    MonotonicTimestamp {
        nanos_since_run_start: sequence * 100,
    }
}

fn provision_pool(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> (
    ResourceTransaction<TestDriver, TransactionCommitted>,
    Arc<TestRuntime>,
    ResourcePoolEvidence,
    Vec<ResourcePoolEvent>,
) {
    let (transaction, _, runtime) = transaction(
        plan,
        &format!("run.provision.{suffix}"),
        &format!("transaction.provision.{suffix}"),
        &format!("request.provision.{suffix}"),
        CommitBehavior::Valid,
    );
    let reserved = transaction.reserve().unwrap();
    let evidence =
        ResourcePoolEvidence::from_external(topology, reserved.admission(), reserved.identity())
            .unwrap();
    let opened = ResourcePoolEvent::opened(1, pool_timestamp(1), &evidence).unwrap();
    let reserve_event = ResourcePoolEvent::transition(
        2,
        pool_timestamp(2),
        &evidence,
        reserved.receipts().last().unwrap(),
        reserved.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    let committed = reserved.commit().unwrap();
    let commit_event = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &evidence,
        committed.receipts().last().unwrap(),
        committed.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    (
        committed,
        runtime,
        evidence,
        vec![opened, reserve_event, commit_event],
    )
}

struct ProvisionedRuntimePool {
    resources: Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: Arc<TestRuntime>,
    evidence: ResourcePoolEvidence,
    journal: Vec<ResourcePoolEvent>,
    committed_snapshot: ResourceLedgerSnapshot,
}

fn provision_runtime_pool(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> ProvisionedRuntimePool {
    let (committed, runtime, evidence, journal) = provision_pool(plan, topology, suffix);
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
    let committed_snapshot = committed.ledger_snapshot();
    let resources = match committed.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("event plan runtime handoff failed: {}", failure.error()),
    };
    ProvisionedRuntimePool {
        resources,
        runtime,
        evidence,
        journal,
        committed_snapshot,
    }
}

fn provision_no_static_plan_runtime(
    plan: &ExecutionPlan,
) -> (Arc<PlanRuntimeResources<TestRuntime>>, Arc<TestRuntime>) {
    let (driver, _) = driver(plan, CommitBehavior::Valid);
    let runtime = Arc::clone(driver.runtime());
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(driver.runtime()),
            id("request.event.no-static.provision"),
        )
        .unwrap()
        .into_parts();
    let resources = match provisioning {
        StaticProvisioning::NoStatic(no_static) => no_static.into_plan_runtime(),
        StaticProvisioning::Required(_) => {
            panic!("no-static event fixture unexpectedly requires static provisioning")
        }
    };
    (resources, runtime)
}

fn close_plan_runtime(
    resources: Arc<PlanRuntimeResources<TestRuntime>>,
) -> PlanRuntimeCloseReceipt {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("event plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("event plan runtime close failed: {:?}", failure.failure()),
    }
}

struct SequenceEvidence {
    active: TrustedActiveSequenceBinding,
    completed: TrustedCompletedSequenceBinding,
    submissions: Vec<SubmittedOperationReceipt>,
    completions: Vec<OperationCompletionReceipt>,
}

struct FailureSequenceEvidence {
    active: TrustedActiveSequenceBinding,
    completed: Option<TrustedCompletedSequenceBinding>,
    aborted: Option<TrustedAbortedSequenceBinding>,
    submissions: Vec<SubmittedOperationReceipt>,
    completions: Vec<OperationCompletionReceipt>,
    journal: Vec<ExecutionEvent>,
}

#[derive(Clone, Copy)]
enum ReplayTerminalFixtureMode {
    ContractFailed,
    Drained,
    Quarantined,
    SubmissionIndeterminateDrained,
}

struct ReplayTerminalFailureEvidence {
    sequence: FailureSequenceEvidence,
    drains: Vec<CompletionDrainReceipt>,
    quarantines: Vec<CompletionQuarantineReceipt>,
}

fn suppress_expected_panic_hook<T>(operation: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    std::panic::set_hook(previous);
    match outcome {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn logical_resources(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = plan_resources.trusted_runtime_binding().unwrap();
    let mut request_maintenance_attempts = 0;
    let request_resources = loop {
        match binding
            .try_admit_request(request.clone(), id(run_id), id(request_id))
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(resources) => break resources,
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    request_maintenance_attempts < MAX_EVENT_MAINTENANCE_ATTEMPTS,
                    "event request admission did not converge after bounded maintenance"
                );
                request_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            RequestResourceAdmissionDecision::Deferred(_) => {
                panic!("event fixture request admission deferred")
            }
            RequestResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event fixture request admission rejected")
            }
        }
    };
    let sequence = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut sequence_maintenance_attempts = 0;
    loop {
        match request_resources
            .try_admit_sequence(sequence.clone())
            .unwrap()
        {
            SequenceResourceAdmissionDecision::Admitted(resources) => break resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    sequence_maintenance_attempts < MAX_EVENT_MAINTENANCE_ATTEMPTS,
                    "event sequence admission did not converge after bounded maintenance"
                );
                sequence_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("event fixture sequence admission deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event fixture sequence admission rejected")
            }
        }
    }
}

fn begin_single_participant_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=MAX_EVENT_MAINTENANCE_ATTEMPTS {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred)
                if attempt < MAX_EVENT_MAINTENANCE_ATTEMPTS =>
            {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            StepResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("event step backing did not converge after bounded maintenance")
            }
            StepResourceAdmissionDecision::Deferred(_) => {
                panic!("event single-participant step unexpectedly deferred")
            }
            StepResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event single-participant step unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded event step admission always returns or panics")
}

fn admit_single_participant_invocation(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=MAX_EVENT_MAINTENANCE_ATTEMPTS {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => return invocation,
            InvocationResourceAdmissionDecision::BackingDeferred(deferred)
                if attempt < MAX_EVENT_MAINTENANCE_ATTEMPTS =>
            {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            InvocationResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("event invocation backing did not converge after bounded maintenance")
            }
            InvocationResourceAdmissionDecision::Deferred(_) => {
                panic!("event single-participant invocation unexpectedly deferred")
            }
            InvocationResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event single-participant invocation unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded event invocation admission always returns or panics")
}

fn execute_sequence(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    frames: u64,
) -> SequenceEvidence {
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let plan = resolved.execution_plan();
    let mut submissions = Vec::new();
    let mut completions = Vec::new();
    let mut sequence = 3_u64;
    let mut invocation = 1_u64;
    for frame in 1..=frames {
        let step = begin_single_participant_step(plan_resources, &batch);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        assert_eq!(frame_id.get(), frame);
        sequence += 1;
        for node_index in 0..plan.payload().nodes().len() {
            sequence += 1;
            let operation_event = node_event(
                plan,
                &active,
                active.run_id(),
                active.request_id(),
                sequence,
                frame,
                invocation,
                node_index,
                ExecutionEventKind::OperationSubmitted,
            );
            let node = &plan.payload().nodes()[node_index];
            let provider = registry.bind(resolved, node.id()).unwrap();
            let completion = encode_and_submit_single(
                &provider,
                resolved,
                operation_event.identity(),
                &frame_id,
                &NodeInvocationId::try_from(invocation).unwrap(),
                node.id(),
                &active,
                admit_single_participant_invocation(plan_resources, &step, node.id()),
                &lane,
                &reaper,
            )
            .unwrap();
            let submission = completion.receipt().clone();
            let completion = match completion.poll().unwrap() {
                CompletionObservation::Terminal(receipt) => receipt,
                _ => panic!("event fixture operation did not reach a terminal fence"),
            };
            assert_eq!(completion.submission(), &submission);
            submissions.push(submission);
            completions.push(completion);
            sequence += 2;
            invocation += 1;
        }
        step.try_retire_normal().unwrap();
        sequence += 1;
    }
    assert_eq!(reaper.retained_count(), 0);
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    }
}

fn execute_failure_then_complete(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    fail_fence: bool,
) -> FailureSequenceEvidence {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    if fail_fence {
        runtime.fail_next_fence();
    }
    let submission = encode_and_submit_single(
        &provider,
        resolved,
        operation_event.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        plan.payload().nodes()[0].id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, plan.payload().nodes()[0].id()),
        &lane,
        &reaper,
    )
    .unwrap();
    let submitted_operation = submission.receipt().clone();
    let operation_completion = match submission.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("event failure fixture operation did not reach a terminal fence"),
    };
    let failure = match operation_completion.disposition() {
        OperationCompletionDisposition::FailedButQuiescent(failures) => {
            failures.first().expect("one participant failure").clone()
        }
        OperationCompletionDisposition::Succeeded => IdentifiedFailure::new(
            submitted_operation.participants()[0].identity().clone(),
            FailureEnvelope::new(
                FailureDomain::Device,
                "synthetic_success_reversal",
                "successful fence was incorrectly reported as failed",
                false,
            )
            .unwrap(),
        )
        .unwrap(),
        disposition => panic!("event failure fixture received {disposition:?}"),
    };
    assert_eq!(reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    let failure_event = operation_failure_event(&failure, 6);
    let journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
        operation_event,
        failure_event,
        sequence_completed_event(plan, &active, &completed, 7),
        request_failed_terminal_event(plan, &active, Some(&completed), None, &failure, 8),
    ];
    FailureSequenceEvidence {
        active,
        completed: Some(completed),
        aborted: None,
        submissions: vec![submitted_operation],
        completions: vec![operation_completion],
        journal,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_terminal_failure_then_complete(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    mode: ReplayTerminalFixtureMode,
) -> ReplayTerminalFailureEvidence {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let had_submission_fence = !matches!(
        mode,
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained
    );
    let operation_sequence = if had_submission_fence { 5 } else { 4 };
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        operation_sequence,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    match mode {
        ReplayTerminalFixtureMode::ContractFailed => runtime.contract_fail_next_fence(),
        ReplayTerminalFixtureMode::Drained | ReplayTerminalFixtureMode::Quarantined => {
            runtime.make_next_fence_indeterminate();
        }
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => runtime.panic_next_submit(),
    }
    let dispatch = suppress_expected_panic_hook(|| {
        encode_and_submit_single(
            &provider,
            resolved,
            operation_event.identity(),
            &frame_id,
            &NodeInvocationId::try_from(1).unwrap(),
            plan.payload().nodes()[0].id(),
            &active,
            admit_single_participant_invocation(
                plan_resources,
                &step,
                plan.payload().nodes()[0].id(),
            ),
            &lane,
            &reaper,
        )
    });

    let mut submissions = Vec::new();
    let mut completions = Vec::new();
    let mut drains = Vec::new();
    let mut quarantines = Vec::new();
    let terminal_identity = match mode {
        ReplayTerminalFixtureMode::ContractFailed => {
            let handle = dispatch.unwrap();
            submissions.push(handle.receipt().clone());
            let receipt = match handle.poll().unwrap() {
                CompletionObservation::Terminal(receipt) => receipt,
                _ => panic!("contract terminal fixture did not complete"),
            };
            assert!(matches!(
                receipt.disposition(),
                OperationCompletionDisposition::ContractFailedButQuiescent(_)
            ));
            let identity = receipt.submission().participants()[0].identity().clone();
            completions.push(receipt);
            runtime.reset_stream_failure();
            identity
        }
        ReplayTerminalFixtureMode::Drained | ReplayTerminalFixtureMode::Quarantined => {
            let handle = dispatch.unwrap();
            let slot_id = handle.slot_id();
            submissions.push(handle.receipt().clone());
            assert!(matches!(
                handle.poll().unwrap(),
                CompletionObservation::Indeterminate(_)
            ));
            assert!(matches!(
                handle.wait().unwrap(),
                CompletionObservation::Indeterminate(_)
            ));
            if matches!(mode, ReplayTerminalFixtureMode::Quarantined) {
                runtime.set_synchronize_fails(true);
            }
            match reaper.recover_slot_by_draining_lane(slot_id).unwrap() {
                CompletionRecoveryOutcome::Drained(receipt) => {
                    assert!(matches!(mode, ReplayTerminalFixtureMode::Drained));
                    drains.push(receipt);
                }
                CompletionRecoveryOutcome::Quarantined(receipt) => {
                    assert!(matches!(mode, ReplayTerminalFixtureMode::Quarantined));
                    quarantines.push(receipt);
                    runtime.set_synchronize_fails(false);
                    assert!(matches!(
                        reaper.recover_slot_by_draining_lane(slot_id).unwrap(),
                        CompletionRecoveryOutcome::Drained(_)
                    ));
                }
            }
            submissions[0].participants()[0].identity().clone()
        }
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => {
            let recovery = match dispatch {
                Err(OperationDispatchError::SubmissionIndeterminate { recovery }) => recovery,
                _ => panic!("submit panic fixture did not retain recovery authority"),
            };
            let receipt = match recovery.recover_by_draining_lane().unwrap() {
                CompletionRecoveryOutcome::Drained(receipt) => receipt,
                CompletionRecoveryOutcome::Quarantined(_) => {
                    panic!("submit panic fixture unexpectedly quarantined")
                }
            };
            let identity = receipt.batch_identity().participants()[0]
                .identity()
                .clone();
            drains.push(receipt);
            identity
        }
    };
    runtime.set_synchronize_fails(false);
    runtime.reset_stream_failure();
    assert_eq!(reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    let failure = IdentifiedFailure::new(
        terminal_identity,
        FailureEnvelope::new(
            FailureDomain::Device,
            match mode {
                ReplayTerminalFixtureMode::ContractFailed => "contract_terminal_failure",
                ReplayTerminalFixtureMode::Drained => "drained_terminal_failure",
                ReplayTerminalFixtureMode::Quarantined => "quarantined_terminal_failure",
                ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => {
                    "submission_indeterminate_terminal_failure"
                }
            },
            "operation did not produce a successful replay terminal",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let failure_sequence = if had_submission_fence { 6 } else { 5 };
    let sequence_disposition = failure_sequence + 1;
    let request_terminal = failure_sequence + 2;
    let mut journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
    ];
    if had_submission_fence {
        journal.push(operation_event);
    }
    journal.push(operation_failure_event(&failure, failure_sequence));
    journal.push(sequence_completed_event(
        plan,
        &active,
        &completed,
        sequence_disposition,
    ));
    journal.push(request_failed_terminal_event(
        plan,
        &active,
        Some(&completed),
        None,
        &failure,
        request_terminal,
    ));
    ReplayTerminalFailureEvidence {
        sequence: FailureSequenceEvidence {
            active,
            completed: Some(completed),
            aborted: None,
            submissions,
            completions,
            journal,
        },
        drains,
        quarantines,
    }
}

fn emit_pre_active_prefix(
    emitter: &mut ExecutionEventEmitter<'_>,
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
) {
    let accepted = accepted_event(active.run_id(), active.request_id());
    emitter
        .emit(
            &accepted,
            &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        )
        .unwrap();
    let planned = plan_event(plan, active.run_id(), active.request_id());
    emitter
        .emit(
            &planned,
            &TrustedExecutionEventContext::bound(
                active.run_id(),
                active.request_id(),
                &TrustedExecutionTopology::from_plan(plan).unwrap(),
            ),
        )
        .unwrap();
}

fn live_witness_emitter_contract(
    passed: &mut usize,
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) {
    let plan = resolved.execution_plan();
    let topology = TrustedExecutionTopology::from_plan(plan).unwrap();

    let resources = logical_resources(plan_resources, "run.emitter.live", "request.emitter.live");
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &frame,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_ok()
            && emitter.cursor().last_sequence() == 3
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameStarted),
    );
    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.completed-stale",
        "request.emitter.completed-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &frame,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && !emitter.sink_failed()
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.aborted-stale",
        "request.emitter.aborted-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &frame,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.dropped-stale",
        "request.emitter.dropped-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    drop(session);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &frame,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.cancel-progress",
        "request.emitter.cancel-progress",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    for event in [
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
    ] {
        emitter
            .emit(
                &event,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .unwrap();
    }
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let first_operation = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let first_node = &plan.payload().nodes()[0];
    let first_provider = registry.bind(resolved, first_node.id()).unwrap();
    let first_handle = encode_and_submit_single(
        &first_provider,
        resolved,
        first_operation.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        first_node.id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, first_node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    emitter
        .emit(
            &first_operation,
            &TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                first_handle.receipt(),
            ),
        )
        .unwrap();
    let first_completion = match first_handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("first node did not complete: {other:?}"),
    };
    let first_retired = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        6,
        1,
        1,
        0,
        ExecutionEventKind::NodeRetired,
    );
    emitter
        .emit(
            &first_retired,
            &TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &first_completion.participants()[0],
            ),
        )
        .unwrap();
    let second_started = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        2,
        1,
        ExecutionEventKind::NodeStarted,
    );
    emitter
        .emit(
            &second_started,
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    let second_operation = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        8,
        1,
        2,
        1,
        ExecutionEventKind::OperationSubmitted,
    );
    let second_node = &plan.payload().nodes()[1];
    let second_provider = registry.bind(resolved, second_node.id()).unwrap();
    let second_handle = encode_and_submit_single(
        &second_provider,
        resolved,
        second_operation.identity(),
        &frame_id,
        &NodeInvocationId::try_from(2).unwrap(),
        second_node.id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, second_node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    session.request_cancel().unwrap();
    emitter
        .emit(
            &second_operation,
            &TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                second_handle.receipt(),
            ),
        )
        .unwrap();
    let second_completion = match second_handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("second node did not complete: {other:?}"),
    };
    let second_retired = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        9,
        1,
        2,
        1,
        ExecutionEventKind::NodeRetired,
    );
    emitter
        .emit(
            &second_retired,
            &TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &second_completion.participants()[0],
            ),
        )
        .unwrap();
    let frame_completed = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        10,
        1,
        ExecutionEventKind::FrameCompleted,
    );
    emitter
        .emit(
            &frame_completed,
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    check(
        passed,
        emitter.cursor().last_sequence() == 10
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameCompleted),
    );
    step.try_abort().unwrap();
    session.try_abort().unwrap();
    drop(emitter);
    drop(second_handle);
    drop(first_handle);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.cancel-start",
        "request.emitter.cancel-start",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    emitter
        .emit(
            &frame,
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    session.request_cancel().unwrap();
    let node = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        1,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &node,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 3
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameStarted),
    );
    step.try_abort().unwrap();
    session.try_abort().unwrap();
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.legacy-rejected",
        "request.emitter.legacy-rejected",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let active = TrustedActiveSequenceBinding::from_permit(&permit).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                &frame,
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    let _legacy_completion = permit.synchronize().unwrap().complete().unwrap();
    drop(emitter);
    drop(stream);
    drop(resources);
}

fn execute_failure_then_abort(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    pool_journal: Vec<ResourcePoolEvent>,
    pool_evidence: ResourcePoolEvidence,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
) -> (
    FailureSequenceEvidence,
    ResourcePoolEvidence,
    Vec<ResourcePoolEvent>,
) {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    runtime.fail_next_fence();
    let submission = encode_and_submit_single(
        &provider,
        resolved,
        operation_event.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        plan.payload().nodes()[0].id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, plan.payload().nodes()[0].id()),
        &lane,
        &reaper,
    )
    .unwrap();
    let submitted_operation = submission.receipt().clone();
    let operation_completion = match submission.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("event abort fixture operation did not reach a terminal fence"),
    };
    let failure = match operation_completion.disposition() {
        OperationCompletionDisposition::FailedButQuiescent(failures) => {
            failures.first().expect("one participant failure").clone()
        }
        disposition => panic!("event abort fixture received {disposition:?}"),
    };
    assert_eq!(reaper.retained_count(), 0);
    let failure_event = operation_failure_event(&failure, 6);
    step.try_abort().unwrap();
    let abort_receipt = session.try_abort().unwrap();
    let aborted =
        TrustedAbortedSequenceBinding::from_session_receipt(&abort_receipt, &active).unwrap();
    let journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
        operation_event,
        failure_event,
        sequence_aborted_event(plan, &active, &aborted, 7),
        request_failed_terminal_event(plan, &active, None, Some(&aborted), &failure, 8),
    ];
    (
        FailureSequenceEvidence {
            active,
            completed: None,
            aborted: Some(aborted),
            submissions: vec![submitted_operation],
            completions: vec![operation_completion],
            journal,
        },
        pool_evidence,
        pool_journal,
    )
}

fn failure_recovery_pair(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> (
    ResourcePoolEvidence,
    ResourcePoolEvent,
    ResourcePoolEvent,
    ResourcePoolEvent,
    ResourceFailureReceipt,
    ResourceFailureReceipt,
) {
    let (transaction, _, _) = transaction(
        plan,
        &format!("run.failure.{suffix}"),
        &format!("transaction.failure.{suffix}"),
        &format!("request.failure.{suffix}"),
        CommitBehavior::InvalidFirst,
    );
    let reserved = transaction.reserve().unwrap();
    let evidence =
        ResourcePoolEvidence::from_external(topology, reserved.admission(), reserved.identity())
            .unwrap();
    let opened = ResourcePoolEvent::opened(1, pool_timestamp(1), &evidence).unwrap();
    let reserve_event = ResourcePoolEvent::transition(
        2,
        pool_timestamp(2),
        &evidence,
        reserved.receipts().last().unwrap(),
        reserved.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    let recovery_owner = match reserved.commit() {
        Err(ResourceCommitTransitionError::Recoverable(recovery)) => recovery,
        Err(ResourceCommitTransitionError::Poisoned(_)) => {
            panic!("expected recoverable commit failure")
        }
        Ok(_) => panic!("expected recoverable commit failure"),
    };
    let anchor = recovery_owner.failure().clone();
    let failed = ResourcePoolEvent::failed(3, pool_timestamp(3), &evidence, &anchor).unwrap();
    let recovered_transaction = recovery_owner.recover().unwrap();
    let recovery = recovered_transaction
        .recovery_history()
        .last()
        .unwrap()
        .clone();
    let _rolled_back = recovered_transaction.rollback().unwrap();
    (evidence, opened, reserve_event, failed, anchor, recovery)
}

fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}

#[test]
fn vnext_event_replay_v5_contract() {
    const EXPECTED_CASES: usize = 161;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("v4", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    check(
        &mut passed,
        plan.payload().nodes().len() == 2
            && plan.payload().nodes()[1].dependencies() == [id("node.first")],
    );
    check(&mut passed, ExecutionFrameId::try_from(0).is_err());
    check(&mut passed, NodeInvocationId::try_from(0).is_err());
    check(
        &mut passed,
        topology.device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );

    let resolved = resolved_model_plan(&plan, "v4", &operation_registry);
    let ProvisionedRuntimePool {
        resources: plan_resources,
        runtime: plan_runtime,
        evidence: pool_evidence,
        journal: pool_journal,
        committed_snapshot,
    } = provision_runtime_pool(&plan, &topology, "v4");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.request.one",
        "request.one",
        2,
    );
    check(
        &mut passed,
        active.runtime_implementation_fingerprint()
            == topology.device_runtime_implementation_fingerprint(),
    );
    check(
        &mut passed,
        !serde_json::to_string(&active)
            .unwrap()
            .contains("runtime_type"),
    );
    let SequenceEvidence {
        active: active_two,
        completed: completed_two,
        submissions: submissions_two,
        completions: completions_two,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.request.two",
        "request.two",
        1,
    );

    let completed_failure = execute_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.completed",
        "request.failure.completed",
        true,
    );
    check(
        &mut passed,
        observe_failure_journal(&completed_failure, &topology)
            .unwrap()
            .is_terminal(),
    );
    let completed_failure_first = match completed_failure.journal[5].detail() {
        ExecutionEventDetail::Failure(failure) => failure,
        _ => unreachable!(),
    };
    check(
        &mut passed,
        matches!(
            completed_failure.journal.last().unwrap().detail(),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint
            } if first_failure_fingerprint == &completed_failure_first.fingerprint()
        ),
    );

    let succeeded_completion_failure = execute_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.synthetic",
        "request.failure.synthetic",
        false,
    );
    plan_runtime.fail_next_fence();
    let failed_completion_success = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.success.with-failed-completion",
        "request.success.with-failed-completion",
        3,
    );
    let failed_completion_success_journal = request_journal(
        &plan,
        &failed_completion_success.active,
        &failed_completion_success.completed,
        &failed_completion_success.submissions,
        &failed_completion_success.completions,
        3,
    );
    let contract_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.contract-terminal",
        "request.failure.contract-terminal",
        ReplayTerminalFixtureMode::ContractFailed,
    );
    let drained_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.drained-terminal",
        "request.failure.drained-terminal",
        ReplayTerminalFixtureMode::Drained,
    );
    let quarantined_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.quarantined-terminal",
        "request.failure.quarantined-terminal",
        ReplayTerminalFixtureMode::Quarantined,
    );
    let no_fence_drained_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.no-fence-drained-terminal",
        "request.failure.no-fence-drained-terminal",
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained,
    );
    live_witness_emitter_contract(
        &mut passed,
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
    );

    let ProvisionedRuntimePool {
        resources: abort_plan_resources,
        runtime: abort_runtime,
        evidence: abort_pool_evidence,
        journal: abort_pool_prefix,
        committed_snapshot: abort_committed_snapshot,
    } = provision_runtime_pool(&plan, &topology, "abort-v5");
    let (aborted_failure, abort_pool_evidence, abort_pool_journal) = execute_failure_then_abort(
        &abort_plan_resources,
        &abort_runtime,
        abort_pool_prefix,
        abort_pool_evidence,
        &resolved,
        &operation_registry,
        "run.failure.aborted",
        "request.failure.aborted",
    );
    let close_receipt = close_plan_runtime(plan_resources);
    let abort_close_receipt = close_plan_runtime(abort_plan_resources);
    check(
        &mut passed,
        observe_failure_journal(&aborted_failure, &topology)
            .unwrap()
            .is_terminal(),
    );
    check(
        &mut passed,
        aborted_failure.aborted.as_ref().unwrap().disposition()
            == ActiveSequenceAbortDisposition::SequenceSessionTerminalized,
    );
    let aborted_failure_first = match aborted_failure.journal[5].detail() {
        ExecutionEventDetail::Failure(failure) => failure,
        _ => unreachable!(),
    };
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(
            &serde_json::to_vec(&completed_failure.journal[5]).unwrap(),
        )
        .unwrap()
        .revalidate(&TrustedExecutionEventContext::failure(
            completed_failure.active.run_id(),
            completed_failure.active.request_id(),
            Some(&topology),
            Some(&completed_failure.active),
            completed_failure_first,
        ))
        .unwrap()
            == completed_failure.journal[5],
    );
    let failure_wire = serde_json::to_value(&completed_failure.journal[5]).unwrap();
    let mut event_unknown_top = failure_wire.clone();
    event_unknown_top["unknown_top"] = json!(true);
    let mut event_unknown_identity = failure_wire.clone();
    event_unknown_identity["identity"]["unknown_identity"] = json!(true);
    let mut event_unknown_detail = failure_wire.clone();
    event_unknown_detail["detail"]["failure"]["failure"]["unknown_nested"] = json!(true);
    for wire in [
        event_unknown_top,
        event_unknown_identity,
        event_unknown_detail,
    ] {
        check(
            &mut passed,
            ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).is_err(),
        );
    }
    let mut event_unknown_variant =
        serde_json::to_value(completed_failure.journal.last().unwrap()).unwrap();
    event_unknown_variant["detail"]["failure_terminal"]["extra"] = json!(true);
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(&event_unknown_variant).unwrap())
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(
            &serde_json::to_vec(&completed_failure.journal[5]).unwrap(),
        )
        .unwrap()
        .revalidate(&TrustedExecutionEventContext::failure(
            completed_failure.active.run_id(),
            completed_failure.active.request_id(),
            Some(&topology),
            Some(&completed_failure.active),
            aborted_failure_first,
        ))
        .is_err(),
    );

    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 2);
    let cursor = observe_journal(
        &journal,
        &topology,
        &active,
        &completed,
        &submissions,
        &completions,
    )
    .unwrap();
    check(
        &mut passed,
        cursor.is_terminal() && cursor.completed_frames() == 2,
    );
    let invocations = journal
        .iter()
        .filter(|event| event.kind() == ExecutionEventKind::NodeStarted)
        .map(|event| {
            (
                event.identity().parts().frame_id.unwrap().get(),
                event.identity().parts().node_id.clone().unwrap(),
                event.identity().parts().node_invocation_id.unwrap().get(),
            )
        })
        .collect::<Vec<_>>();
    check(
        &mut passed,
        invocations.len() == 4
            && invocations[0].1 == invocations[2].1
            && invocations[0].2 == 1
            && invocations[2].2 == 3,
    );

    let mut frame_jump =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..2] {
        frame_jump
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let jump = frame_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        2,
        ExecutionEventKind::FrameStarted,
    );
    check(
        &mut passed,
        frame_jump
            .observe_against(
                &jump,
                &event_context(
                    &jump,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && frame_jump.last_sequence() == 2,
    );

    let mut incomplete =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        incomplete
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let premature = frame_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        ExecutionEventKind::FrameCompleted,
    );
    check(
        &mut passed,
        incomplete
            .observe_against(
                &premature,
                &event_context(
                    &premature,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && incomplete.last_sequence() == 3,
    );

    let mut cross_frame =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..11] {
        cross_frame
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let dependency_from_prior_frame = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        12,
        2,
        3,
        1,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        cross_frame
            .observe_against(
                &dependency_from_prior_frame,
                &event_context(
                    &dependency_from_prior_frame,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err(),
    );

    let mut duplicate_invocation =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..6] {
        duplicate_invocation
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let duplicate = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        1,
        1,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        duplicate_invocation
            .observe_against(
                &duplicate,
                &event_context(
                    &duplicate,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && duplicate_invocation.last_sequence() == 6,
    );

    let mut invocation_gap =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        invocation_gap
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let gap = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        2,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        invocation_gap
            .observe_against(
                &gap,
                &event_context(
                    &gap,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && invocation_gap.last_sequence() == 3,
    );

    let mut duplicate_node =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..6] {
        duplicate_node
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let repeated_completed_node = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        2,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        duplicate_node
            .observe_against(
                &repeated_completed_node,
                &event_context(
                    &repeated_completed_node,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && duplicate_node.last_sequence() == 6,
    );

    let node_started = &journal[3];
    let mut node_prefix =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        node_prefix
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    for (field, replacement) in [
        ("resource_pool_id", json!(999_999_u64)),
        ("resource_pool_identity_fingerprint", json!(sha('0'))),
        (
            "activation_epoch",
            json!(active.activation_epoch().saturating_add(1)),
        ),
        ("runtime_implementation_fingerprint", json!(sha('0'))),
        ("active_sequence_fingerprint", json!(sha('0'))),
        ("frame_id", json!(2)),
        ("node_invocation_id", json!(2)),
    ] {
        let mut wire = serde_json::to_value(node_started).unwrap();
        wire["identity"][field] = replacement;
        let decoded =
            ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).unwrap();
        let mut tampered_cursor = node_prefix.clone();
        let revalidated = decoded.revalidate(&TrustedExecutionEventContext::active(
            active.run_id(),
            active.request_id(),
            &topology,
            &active,
        ));
        let rejected = match revalidated {
            Err(_) => true,
            Ok(tampered) => tampered_cursor
                .observe_against(
                    &tampered,
                    &TrustedExecutionEventContext::active(
                        active.run_id(),
                        active.request_id(),
                        &topology,
                        &active,
                    ),
                )
                .is_err(),
        };
        let mut valid_cursor = node_prefix.clone();
        let rejected = rejected
            && tampered_cursor.last_sequence() == 3
            && valid_cursor
                .observe_against(
                    node_started,
                    &event_context(
                        node_started,
                        &topology,
                        &active,
                        &completed,
                        &submissions,
                        &completions,
                    ),
                )
                .is_ok();
        assert!(
            rejected,
            "wire tamper `{field}` was accepted or mutated cursor"
        );
        passed += 1;
    }
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(node_started).unwrap()).unwrap();
    let unchanged = node_prefix.clone();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::active(
                active_two.run_id(),
                active_two.request_id(),
                &topology,
                &active_two,
            ))
            .is_err()
            && unchanged.last_sequence() == 3,
    );

    let operation_submitted = &journal[4];
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(operation_submitted).unwrap())
            .unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &submissions[0],
            ))
            .unwrap()
            == *operation_submitted,
    );
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(operation_submitted).unwrap())
            .unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &submissions_two[0],
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&vec![b' '; MAX_EXECUTION_EVENT_WIRE_BYTES + 1]).is_err(),
    );
    let mut plan_wire = serde_json::to_value(&journal[1]).unwrap();
    plan_wire["identity"]["runtime_implementation_fingerprint"] = Value::Null;
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(&plan_wire).unwrap())
            .unwrap()
            .revalidate(&TrustedExecutionEventContext::bound(
                active.run_id(),
                active.request_id(),
                &topology,
            ))
            .is_err(),
    );
    let mut device_parts = base_parts(
        active.run_id(),
        active.request_id(),
        1,
        "span.device-failure",
        None,
    );
    device_parts.device_id = Some(plan.payload().device_id().clone());
    check(
        &mut passed,
        ExecutionIdentityEnvelope::new(device_parts.clone()).is_err(),
    );
    device_parts.runtime_implementation_fingerprint = Some(
        plan.payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
    );
    let device_identity = ExecutionIdentityEnvelope::new(device_parts).unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(
            device_identity,
            FailureEnvelope::new(
                FailureDomain::Device,
                "device_failed",
                "device failure",
                false,
            )
            .unwrap(),
        )
        .is_ok(),
    );

    let mut missing_submission =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..4] {
        missing_submission
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let node_without_submission = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::NodeRetired,
    );
    check(
        &mut passed,
        missing_submission
            .observe_against(
                &node_without_submission,
                &event_context(
                    &node_without_submission,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && missing_submission.last_sequence() == 4,
    );

    let sequence_completed = &journal[journal.len() - 2];
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(sequence_completed).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &completed,
            ))
            .unwrap()
            == *sequence_completed,
    );
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(sequence_completed).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &completed_two,
            ))
            .is_err(),
    );
    let mut before_sync =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..journal.len() - 2] {
        before_sync
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let premature_terminal = request_completed_event(
        &plan,
        &active,
        &completed,
        active.run_id(),
        active.request_id(),
        before_sync.last_sequence() + 1,
    );
    check(
        &mut passed,
        before_sync
            .observe_against(
                &premature_terminal,
                &TrustedExecutionEventContext::completed(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                    &completed,
                ),
            )
            .is_err()
            && !before_sync.is_terminal(),
    );

    let mut no_failure_disposition = ExecutionEventCursor::new(
        completed_failure.active.run_id().clone(),
        completed_failure.active.request_id().clone(),
    );
    for event in &completed_failure.journal[..6] {
        let context = match event.kind() {
            ExecutionEventKind::RequestAccepted => TrustedExecutionEventContext::pre_plan(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
            ),
            ExecutionEventKind::PlanBuilt => TrustedExecutionEventContext::bound(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                &topology,
            ),
            ExecutionEventKind::OperationSubmitted => {
                TrustedExecutionEventContext::operation_submitted(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    &completed_failure.submissions[0],
                )
            }
            ExecutionEventKind::FailureObserved => TrustedExecutionEventContext::failure(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                Some(&topology),
                Some(&completed_failure.active),
                completed_failure_first,
            ),
            _ => TrustedExecutionEventContext::active(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                &topology,
                &completed_failure.active,
            ),
        };
        no_failure_disposition
            .observe_against(event, &context)
            .unwrap();
    }
    let terminal_without_disposition = request_failed_terminal_event(
        &plan,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        completed_failure_first,
        7,
    );
    check(
        &mut passed,
        no_failure_disposition
            .observe_against(
                &terminal_without_disposition,
                &TrustedExecutionEventContext::failure_with_disposition(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    completed_failure.completed.as_ref(),
                    None,
                    completed_failure_first,
                ),
            )
            .is_err()
            && !no_failure_disposition.is_terminal(),
    );

    let completed_terminal = completed_failure.journal.last().unwrap();
    let wrong_terminal_fingerprint = ExecutionEvent::new(
        completed_terminal.timestamp(),
        completed_terminal.phase(),
        ExecutionEventKind::RequestFailed,
        completed_terminal.identity().clone(),
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: sha('0'),
        },
    )
    .unwrap();
    let mut terminal_prefix = observe_failure_journal(
        &FailureSequenceEvidence {
            active: completed_failure.active.clone(),
            completed: completed_failure.completed.clone(),
            aborted: None,
            submissions: completed_failure.submissions.clone(),
            completions: completed_failure.completions.clone(),
            journal: completed_failure.journal[..7].to_vec(),
        },
        &topology,
    )
    .unwrap();
    check(
        &mut passed,
        terminal_prefix
            .observe_against(
                &wrong_terminal_fingerprint,
                &TrustedExecutionEventContext::failure_with_disposition(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    completed_failure.completed.as_ref(),
                    None,
                    completed_failure_first,
                ),
            )
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::new(
            MonotonicTimestamp {
                nanos_since_run_start: 80,
            },
            ExecutionPhase::Completion,
            ExecutionEventKind::RequestFailed,
            completed_failure.journal[5].identity().clone(),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint: completed_failure_first.fingerprint(),
            },
        )
        .is_err(),
    );

    let mut non_active_failure =
        ExecutionEventCursor::new(active_two.run_id().clone(), active_two.request_id().clone());
    non_active_failure
        .observe_against(
            &accepted_event(active_two.run_id(), active_two.request_id()),
            &TrustedExecutionEventContext::pre_plan(active_two.run_id(), active_two.request_id()),
        )
        .unwrap();
    non_active_failure
        .observe_against(
            &plan_event(&plan, active_two.run_id(), active_two.request_id()),
            &TrustedExecutionEventContext::bound(
                active_two.run_id(),
                active_two.request_id(),
                &topology,
            ),
        )
        .unwrap();
    let (planning_failure, planning_failure_evidence) =
        planning_failure_event(&plan, active_two.run_id(), active_two.request_id(), 3);
    non_active_failure
        .observe_against(
            &planning_failure,
            &TrustedExecutionEventContext::failure(
                active_two.run_id(),
                active_two.request_id(),
                Some(&topology),
                None,
                &planning_failure_evidence,
            ),
        )
        .unwrap();
    let foreign_disposition = sequence_completed_event(&plan, &active_two, &completed_two, 4);
    check(
        &mut passed,
        non_active_failure
            .observe_against(
                &foreign_disposition,
                &TrustedExecutionEventContext::completed(
                    active_two.run_id(),
                    active_two.request_id(),
                    &topology,
                    &active_two,
                    &completed_two,
                ),
            )
            .is_err(),
    );
    let mut non_active_abort = ExecutionEventCursor::new(
        aborted_failure.active.run_id().clone(),
        aborted_failure.active.request_id().clone(),
    );
    let abort_accepted = accepted_event(
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
    );
    non_active_abort
        .observe_against(
            &abort_accepted,
            &TrustedExecutionEventContext::pre_plan(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
            ),
        )
        .unwrap();
    let abort_plan = plan_event(
        &plan,
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
    );
    non_active_abort
        .observe_against(
            &abort_plan,
            &TrustedExecutionEventContext::bound(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
                &topology,
            ),
        )
        .unwrap();
    let (abort_planning_failure, abort_planning_evidence) = planning_failure_event(
        &plan,
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
        3,
    );
    non_active_abort
        .observe_against(
            &abort_planning_failure,
            &TrustedExecutionEventContext::failure(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
                Some(&topology),
                None,
                &abort_planning_evidence,
            ),
        )
        .unwrap();
    let foreign_abort = sequence_aborted_event(
        &plan,
        &aborted_failure.active,
        aborted_failure.aborted.as_ref().unwrap(),
        4,
    );
    check(
        &mut passed,
        non_active_abort
            .observe_against(
                &foreign_abort,
                &TrustedExecutionEventContext::aborted(
                    aborted_failure.active.run_id(),
                    aborted_failure.active.request_id(),
                    &topology,
                    &aborted_failure.active,
                    aborted_failure.aborted.as_ref().unwrap(),
                ),
            )
            .is_err(),
    );

    let preplan_run: RunId = id("run.failure.preplan");
    let preplan_request: RequestIdentity = id("request.failure.preplan");
    let preplan_identity = ExecutionIdentityEnvelope::new(base_parts(
        &preplan_run,
        &preplan_request,
        1,
        "span.request",
        None,
    ))
    .unwrap();
    let preplan_failure = IdentifiedFailure::new(
        preplan_identity.clone(),
        FailureEnvelope::new(
            FailureDomain::ModelResolution,
            "preplan_failure",
            "preplan failure",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let preplan_terminal = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 10,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestFailed,
        preplan_identity,
        ExecutionEventDetail::Failure(preplan_failure.clone()),
    )
    .unwrap();
    let mut preplan_cursor =
        ExecutionEventCursor::new(preplan_run.clone(), preplan_request.clone());
    check(
        &mut passed,
        preplan_cursor
            .observe_against(
                &preplan_terminal,
                &TrustedExecutionEventContext::failure(
                    &preplan_run,
                    &preplan_request,
                    None,
                    None,
                    &preplan_failure,
                ),
            )
            .is_ok()
            && preplan_cursor.is_terminal(),
    );

    let accepted = &journal[0];
    let mut wrong_cursor =
        ExecutionEventCursor::new(id("run.cursor.wrong"), active.request_id().clone());
    check(
        &mut passed,
        wrong_cursor
            .observe_against(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err()
            && wrong_cursor.last_sequence() == 0,
    );
    let decoded = ExecutionEvent::decode_untrusted(&serde_json::to_vec(accepted).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::pre_plan(
                active.run_id(),
                active.request_id(),
            ))
            .unwrap()
            == *accepted,
    );
    let mut wire = serde_json::to_value(accepted).unwrap();
    wire["identity"]["sequence"] = json!(0);
    let decoded = ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::pre_plan(
                active.run_id(),
                active.request_id(),
            ))
            .is_err(),
    );

    let model_failure = FailureEnvelope::new(
        FailureDomain::ModelResolution,
        "model_resolution_failed",
        "model resolution failed",
        false,
    )
    .unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(accepted.identity().clone(), model_failure).is_ok(),
    );
    let wrong_domain = FailureEnvelope::new(
        FailureDomain::Operation,
        "operation_failed",
        "operation failed",
        false,
    )
    .unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(accepted.identity().clone(), wrong_domain.clone()).is_err(),
    );
    check(
        &mut passed,
        IdentifiedFailure::new(journal[4].identity().clone(), wrong_domain).is_ok(),
    );

    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emitter
        .emit(
            accepted,
            &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        )
        .unwrap();
    check(
        &mut passed,
        sink.kinds.lock().unwrap().as_slice() == [ExecutionEventKind::RequestAccepted],
    );
    check(&mut passed, emitter.cursor().last_sequence() == 1);
    let mut failed_emitter = ExecutionEventEmitter::new(
        &FailingSink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    check(
        &mut passed,
        failed_emitter
            .emit(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err(),
    );
    check(
        &mut passed,
        failed_emitter.cursor().last_sequence() == 0 && failed_emitter.sink_failed(),
    );
    check(
        &mut passed,
        failed_emitter
            .emit(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err(),
    );
    let event_source = include_str!("../src/vnext/event.rs");
    check(
        &mut passed,
        event_source.contains("pub struct EventEmissionPermit<'event>")
            && !event_source.contains("pub fn new_event_emission_permit"),
    );

    let mut pool_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    for event in &pool_journal {
        pool_cursor.observe(event).unwrap();
    }
    check(
        &mut passed,
        pool_cursor.is_open() && pool_cursor.last_sequence() == 3,
    );
    check(&mut passed, cursor.is_terminal() && pool_cursor.is_open());

    let opened_wire =
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[0]).unwrap())
            .unwrap();
    check(
        &mut passed,
        opened_wire
            .revalidate(&TrustedResourcePoolEventContext::opened(&pool_evidence))
            .unwrap()
            == pool_journal[0],
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[0]).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::opened(
                &abort_pool_evidence,
            ))
            .is_err(),
    );
    let (reserve_receipt, reserve_context) = match pool_journal[1].detail() {
        ResourcePoolEventDetail::Transition { receipt, context } => (receipt, context),
        _ => unreachable!(),
    };
    let transition_wire =
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[1]).unwrap())
            .unwrap();
    check(
        &mut passed,
        transition_wire
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &pool_evidence,
                reserve_context,
            ))
            .unwrap()
            == pool_journal[1],
    );
    check(
        &mut passed,
        ResourcePoolEvent::transition(
            2,
            pool_timestamp(2),
            &abort_pool_evidence,
            reserve_receipt,
            reserve_context,
        )
        .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[1]).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &abort_pool_evidence,
                reserve_context,
            ))
            .is_err(),
    );
    let transition_value = serde_json::to_value(&pool_journal[1]).unwrap();
    let mut pool_unknown_top = transition_value.clone();
    pool_unknown_top["unknown_top"] = json!(true);
    let mut pool_unknown_identity = transition_value.clone();
    pool_unknown_identity["identity"]["unknown_identity"] = json!(true);
    let mut pool_unknown_detail = transition_value;
    pool_unknown_detail["detail"]["transition"]["receipt"]["unknown_nested"] = json!(true);
    for wire in [pool_unknown_top, pool_unknown_identity, pool_unknown_detail] {
        check(
            &mut passed,
            ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).is_err(),
        );
    }
    let mut pool_unknown_variant = serde_json::to_value(&pool_journal[1]).unwrap();
    pool_unknown_variant["detail"]["transition"]["extra"] = json!(true);
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_unknown_variant).unwrap())
            .is_err(),
    );
    for (path, replacement) in [
        ("pool_id", json!(999_998_u64)),
        ("pool_identity_fingerprint", json!(sha('0'))),
        ("transaction_id", json!("transaction.wire-tampered")),
    ] {
        let mut wire = serde_json::to_value(&pool_journal[1]).unwrap();
        wire["identity"][path] = replacement;
        check(
            &mut passed,
            ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap())
                .unwrap()
                .revalidate(&TrustedResourcePoolEventContext::transition(
                    &pool_evidence,
                    reserve_context,
                ))
                .is_err(),
        );
    }
    let mut context_tamper = serde_json::to_value(&pool_journal[1]).unwrap();
    context_tamper["detail"]["transition"]["context"]["action"] = json!("commit");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&context_tamper).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &pool_evidence,
                reserve_context,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&vec![b' '; MAX_RESOURCE_POOL_EVENT_WIRE_BYTES + 1])
            .is_err(),
    );
    check(
        &mut passed,
        event_source.contains(
            "#[derive(Debug, Clone, PartialEq, Eq, Serialize)]\npub struct ResourcePoolEvent",
        ) && event_source.contains("pub struct UnvalidatedResourcePoolEvent"),
    );

    let skipped = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut skipped_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    skipped_cursor.observe(&pool_journal[0]).unwrap();
    check(
        &mut passed,
        skipped_cursor.observe(&skipped).is_err() && skipped_cursor.last_sequence() == 1,
    );
    let reused_reserve_receipt = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut receipt_reuse_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    receipt_reuse_cursor.observe(&pool_journal[0]).unwrap();
    receipt_reuse_cursor.observe(&pool_journal[1]).unwrap();
    check(
        &mut passed,
        receipt_reuse_cursor
            .observe(&reused_reserve_receipt)
            .is_err()
            && receipt_reuse_cursor.last_sequence() == 2,
    );
    let timestamp_rewind = ResourcePoolEvent::transition(
        2,
        pool_timestamp(1),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut timestamp_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    timestamp_cursor.observe(&pool_journal[0]).unwrap();
    check(
        &mut passed,
        timestamp_cursor.observe(&timestamp_rewind).is_err()
            && timestamp_cursor.last_sequence() == 1,
    );

    let (mut lease_committed, _, lease_evidence, _) =
        provision_pool(&plan, &topology, "lease-binding");
    let lease_receipt = lease_committed.defer_all().unwrap();
    let lease_context = lease_committed.latest_lease_validation_context().unwrap();
    let lease_event = ResourcePoolEvent::lease_transition(
        4,
        pool_timestamp(4),
        &lease_evidence,
        &lease_receipt,
        lease_context,
    )
    .unwrap();
    check(
        &mut passed,
        lease_event.kind() == ResourcePoolEventKind::ResourceLeaseTransition,
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&lease_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::lease_transition(
                &lease_evidence,
                lease_context,
            ))
            .unwrap()
            == lease_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::lease_transition(
            4,
            pool_timestamp(4),
            &abort_pool_evidence,
            &lease_receipt,
            lease_context,
        )
        .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&lease_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::lease_transition(
                &abort_pool_evidence,
                lease_context,
            ))
            .is_err(),
    );

    let journal_two = request_journal(
        &plan,
        &active_two,
        &completed_two,
        &submissions_two,
        &completions_two,
        1,
    );
    check(
        &mut passed,
        observe_journal(
            &journal_two,
            &topology,
            &active_two,
            &completed_two,
            &submissions_two,
            &completions_two,
        )
        .unwrap()
        .is_terminal(),
    );
    check(
        &mut passed,
        pool_cursor.is_open() && !pool_cursor.is_closed(),
    );
    check(
        &mut passed,
        active.sequence_authority() != active_two.sequence_authority()
            && active.activation_epoch() == active_two.activation_epoch()
            && active.fingerprint() != active_two.fingerprint()
            && active.static_pool_id() == active_two.static_pool_id(),
    );

    let replay_evidence = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    let replay = ReplayIdentity::from_evidence(&replay_evidence).unwrap();
    check(
        &mut passed,
        replay.terminal_identity() == journal.last().unwrap().identity()
            && replay.random_seed() == 9271,
    );
    let unvalidated =
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap()).unwrap();
    check(
        &mut passed,
        unvalidated.revalidate(&replay_evidence).unwrap() == replay,
    );
    check(
        &mut passed,
        matches!(
            completed_failure.completions[0].disposition(),
            OperationCompletionDisposition::FailedButQuiescent(failures)
                if failures.as_slice() == std::slice::from_ref(completed_failure_first)
        ),
    );
    check(
        &mut passed,
        matches!(
            succeeded_completion_failure.completions[0].disposition(),
            OperationCompletionDisposition::Succeeded
        ),
    );
    check(
        &mut passed,
        matches!(
            failed_completion_success.completions[0].disposition(),
            OperationCompletionDisposition::FailedButQuiescent(_)
        ) && failed_completion_success.completions[1..]
            .iter()
            .all(|receipt| {
                matches!(
                    receipt.disposition(),
                    OperationCompletionDisposition::Succeeded
                )
            }),
    );

    let missing_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions[..completions.len() - 1],
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&missing_operation_terminal).is_err(),
    );
    let mut duplicated_completions = completions.clone();
    duplicated_completions.push(completions[0].clone());
    let duplicate_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &duplicated_completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&duplicate_operation_terminal).is_err(),
    );
    let mut unused_completions = completions.clone();
    unused_completions.push(failed_completion_success.completions[4].clone());
    let unused_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &unused_completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&unused_operation_terminal).is_err(),
    );
    let succeeded_but_failed = ReplayEvidence::new(
        &resolved,
        b"successful fence reported failed",
        b"reversal state",
        9271,
        &succeeded_completion_failure.journal,
        &succeeded_completion_failure.active,
        succeeded_completion_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &succeeded_completion_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&succeeded_but_failed).is_err(),
    );
    let failed_but_succeeded = ReplayEvidence::new(
        &resolved,
        b"failed fence reported successful",
        b"reverse state",
        9271,
        &failed_completion_success_journal,
        &failed_completion_success.active,
        Some(&failed_completion_success.completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &failed_completion_success.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&failed_but_succeeded).is_err(),
    );

    let mismatched_failure = IdentifiedFailure::new(
        completed_failure_first.identity().clone(),
        FailureEnvelope::new(
            FailureDomain::Device,
            "different_terminal_failure",
            "journal failure differs from failed fence",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let mut mismatched_failure_journal = completed_failure.journal.clone();
    let observed_failure_event = &completed_failure.journal[5];
    mismatched_failure_journal[5] = ExecutionEvent::new(
        observed_failure_event.timestamp(),
        observed_failure_event.phase(),
        ExecutionEventKind::FailureObserved,
        observed_failure_event.identity().clone(),
        ExecutionEventDetail::Failure(mismatched_failure.clone()),
    )
    .unwrap();
    let terminal_failure_event = completed_failure.journal.last().unwrap();
    *mismatched_failure_journal.last_mut().unwrap() = ExecutionEvent::new(
        terminal_failure_event.timestamp(),
        terminal_failure_event.phase(),
        ExecutionEventKind::RequestFailed,
        terminal_failure_event.identity().clone(),
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: mismatched_failure.fingerprint(),
        },
    )
    .unwrap();
    let mismatched_failed_completion = ReplayEvidence::new(
        &resolved,
        b"mismatched failed completion",
        b"mismatch state",
        9271,
        &mismatched_failure_journal,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&mismatched_failed_completion).is_err(),
    );

    let contract_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"contract terminal request",
        b"contract terminal state",
        9271,
        &contract_terminal_failure.sequence.journal,
        &contract_terminal_failure.sequence.active,
        contract_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &contract_terminal_failure.sequence.completions,
        &contract_terminal_failure.drains,
        &contract_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let contract_terminal_replay =
        ReplayIdentity::from_evidence(&contract_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        contract_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    let drained_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"drained terminal request",
        b"drained terminal state",
        9271,
        &drained_terminal_failure.sequence.journal,
        &drained_terminal_failure.sequence.active,
        drained_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &drained_terminal_failure.sequence.completions,
        &drained_terminal_failure.drains,
        &drained_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let drained_terminal_replay =
        ReplayIdentity::from_evidence(&drained_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        drained_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    let quarantined_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"quarantined terminal request",
        b"quarantined terminal state",
        9271,
        &quarantined_terminal_failure.sequence.journal,
        &quarantined_terminal_failure.sequence.active,
        quarantined_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &quarantined_terminal_failure.sequence.completions,
        &quarantined_terminal_failure.drains,
        &quarantined_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&quarantined_terminal_replay_evidence).is_err(),
    );
    let no_fence_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"no-fence terminal request",
        b"no-fence terminal state",
        9271,
        &no_fence_drained_terminal_failure.sequence.journal,
        &no_fence_drained_terminal_failure.sequence.active,
        no_fence_drained_terminal_failure
            .sequence
            .completed
            .as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &no_fence_drained_terminal_failure.sequence.completions,
        &no_fence_drained_terminal_failure.drains,
        &no_fence_drained_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let no_fence_terminal_replay =
        ReplayIdentity::from_evidence(&no_fence_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        no_fence_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    check(
        &mut passed,
        no_fence_drained_terminal_failure
            .sequence
            .submissions
            .is_empty()
            && no_fence_drained_terminal_failure
                .sequence
                .completions
                .is_empty()
            && no_fence_drained_terminal_failure.drains.len() == 1
            && !no_fence_drained_terminal_failure.drains[0].had_submission_fence()
            && no_fence_drained_terminal_failure
                .sequence
                .journal
                .iter()
                .all(|event| event.kind() != ExecutionEventKind::OperationSubmitted),
    );
    let cross_type_slot_collision = ReplayEvidence::new(
        &resolved,
        b"cross-type slot collision",
        b"cross-type state",
        9271,
        &drained_terminal_failure.sequence.journal,
        &drained_terminal_failure.sequence.active,
        drained_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &contract_terminal_failure.sequence.completions,
        &drained_terminal_failure.drains,
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&cross_type_slot_collision).is_err(),
    );
    let mut terminal_fingerprint_mutation = serde_json::to_value(&drained_terminal_replay).unwrap();
    terminal_fingerprint_mutation["operation_terminal_evidence_fingerprint"] = json!(sha('0'));
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(
            &serde_json::to_vec(&terminal_fingerprint_mutation).unwrap(),
        )
        .unwrap()
        .revalidate(&drained_terminal_replay_evidence)
        .is_err(),
    );
    let mut terminal_count_mutation = serde_json::to_value(&drained_terminal_replay).unwrap();
    terminal_count_mutation["operation_terminal_evidence_count"] = json!(2);
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&terminal_count_mutation).unwrap())
            .unwrap()
            .revalidate(&drained_terminal_replay_evidence)
            .is_err(),
    );
    let wrong_input = ReplayEvidence::new(
        &resolved,
        b"tampered request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_input)
            .is_err(),
    );
    let wrong_state = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"tampered state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_state)
            .is_err(),
    );
    let wrong_seed = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9272,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_seed)
            .is_err(),
    );
    let truncated = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal[..journal.len() - 1],
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&truncated).is_err(),
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&vec![b' '; MAX_REPLAY_IDENTITY_WIRE_BYTES + 1]).is_err(),
    );
    let other_runtime_catalog = catalog();
    let other_operation_registry = make_operation_registry(&other_runtime_catalog);
    let other_plan = execution_plan("v4-other", &other_operation_registry);
    let other_resolved = resolved_model_plan(&other_plan, "v4-other", &other_operation_registry);
    let wrong_plan = ReplayEvidence::new(
        &other_resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_plan).is_err(),
    );
    let missing_completion = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        None,
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&missing_completion).is_err(),
    );
    let wrong_completion = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed_two),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_completion).is_err(),
    );
    let wrong_submissions = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions_two,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_submissions).is_err(),
    );
    let truncated_pool = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal[..2],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&truncated_pool).is_err(),
    );
    let reordered_pool_journal = vec![
        pool_journal[0].clone(),
        pool_journal[2].clone(),
        pool_journal[1].clone(),
    ];
    let reordered_pool = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &reordered_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reordered_pool).is_err(),
    );
    for field in [
        "resolved_plan_fingerprint",
        "request_journal_fingerprint",
        "active_sequence_fingerprint",
        "pool_journal_fingerprint",
    ] {
        let mut value = serde_json::to_value(&replay).unwrap();
        value[field] = json!(sha('0'));
        check(
            &mut passed,
            ReplayIdentity::decode_untrusted(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .revalidate(&replay_evidence)
                .is_err(),
        );
    }

    let completed_failure_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"failed completion request",
        b"failed completion state",
        9271,
        &completed_failure.journal,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    let completed_failure_replay =
        ReplayIdentity::from_evidence(&completed_failure_replay_evidence).unwrap();
    check(
        &mut passed,
        completed_failure_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );

    let aborted_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    let aborted_replay = ReplayIdentity::from_evidence(&aborted_replay_evidence).unwrap();
    check(
        &mut passed,
        aborted_replay.cleanup_status() == ReplayCleanupStatus::SequenceQuiescent,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&aborted_replay).unwrap())
            .unwrap()
            .revalidate(&aborted_replay_evidence)
            .unwrap()
            == aborted_replay,
    );

    let pending_abort_evidence = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..3],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&pending_abort_evidence)
            .unwrap()
            .cleanup_status()
            == ReplayCleanupStatus::CleanupPending,
    );
    let open_pool_clean_abort = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..3],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&open_pool_clean_abort).is_err(),
    );
    let quarantined_but_open_abort = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..abort_pool_journal.len() - 1],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&quarantined_but_open_abort).is_err(),
    );
    let incomplete_pool_pending = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..abort_pool_journal.len() - 1],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&incomplete_pool_pending).is_err(),
    );
    let wrong_abort_pool = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_abort_pool).is_err(),
    );
    let reused_completion_receipt = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reused_completion_receipt).is_err(),
    );
    let reused_submission_receipt = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reused_submission_receipt).is_err(),
    );
    for field in [
        "aborted_sequence_fingerprint",
        "cleanup_status",
        "resource_pool_identity_fingerprint",
    ] {
        let mut wire = serde_json::to_value(&aborted_replay).unwrap();
        wire[field] = if field == "cleanup_status" {
            json!("completed")
        } else {
            json!(sha('0'))
        };
        check(
            &mut passed,
            ReplayIdentity::decode_untrusted(&serde_json::to_vec(&wire).unwrap())
                .unwrap()
                .revalidate(&aborted_replay_evidence)
                .is_err(),
        );
    }

    let (failure_evidence, failure_opened, failure_reserved, failure_event, anchor, recovery) =
        failure_recovery_pair(&plan, &topology, "one");
    let (_, _, _, _, _, wrong_recovery) = failure_recovery_pair(&plan, &topology, "two");
    let mut failure_cursor = ResourcePoolEventCursor::new(failure_evidence.clone());
    failure_cursor.observe(&failure_opened).unwrap();
    failure_cursor.observe(&failure_reserved).unwrap();
    failure_cursor.observe(&failure_event).unwrap();
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&failure_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &anchor,
            ))
            .unwrap()
            == failure_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &failure_evidence, &anchor)
            .is_err(),
    );
    let mut mislabeled_failure = serde_json::to_value(&failure_event).unwrap();
    mislabeled_failure["kind"] = json!("resource_recovery_completed");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&mislabeled_failure).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &anchor,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::failed(3, pool_timestamp(3), &pool_evidence, &anchor).is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&failure_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &pool_evidence,
                &anchor,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(
            4,
            pool_timestamp(4),
            &failure_evidence,
            &wrong_recovery,
        )
        .is_err()
            && failure_cursor.last_sequence() == 3,
    );
    check(
        &mut passed,
        wrong_recovery
            .validate_recovery_continuation(&anchor)
            .is_err(),
    );
    let recovery_event =
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &failure_evidence, &recovery)
            .unwrap();
    failure_cursor.observe(&recovery_event).unwrap();
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&recovery_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &recovery,
            ))
            .unwrap()
            == recovery_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::failed(4, pool_timestamp(4), &failure_evidence, &recovery).is_err(),
    );
    let mut mislabeled_recovery = serde_json::to_value(&recovery_event).unwrap();
    mislabeled_recovery["kind"] = json!("resource_failed");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&mislabeled_recovery).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &recovery,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &pool_evidence, &recovery)
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&recovery_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &pool_evidence,
                &recovery,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        recovery.failure_id() == anchor.failure_id()
            && recovery.recovery_complete()
            && recovery.validate_recovery_continuation(&anchor).is_ok(),
    );

    check(
        &mut passed,
        ResourcePoolEvent::closed(5, pool_timestamp(5), &pool_evidence, &committed_snapshot)
            .is_err(),
    );
    check(
        &mut passed,
        close_receipt.released_static_resources() == committed_snapshot.entries().len(),
    );
    check(
        &mut passed,
        close_receipt.evidence().static_pool_identity()
            == Some(pool_evidence.admission().pool_identity()),
    );
    check(
        &mut passed,
        close_receipt.evidence().static_provisioning_identity()
            == Some(pool_evidence.provisioning_identity()),
    );
    check(
        &mut passed,
        close_receipt.evidence().plan_hash() == pool_evidence.admission().plan_hash(),
    );

    check(
        &mut passed,
        abort_close_receipt.released_static_resources() == abort_committed_snapshot.entries().len(),
    );
    check(
        &mut passed,
        abort_close_receipt.evidence().static_pool_identity()
            == Some(abort_pool_evidence.admission().pool_identity()),
    );
    lease_committed.resume_all().unwrap();
    let _lease_released = lease_committed.release().unwrap();

    no_static_replay_requires_explicit_root_cleanup();

    assert_eq!(passed, EXPECTED_CASES);
    println!("VNEXT EVENT/REPLAY V5 PASS: {passed}/{EXPECTED_CASES}");
}

fn no_static_replay_requires_explicit_root_cleanup() {
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = no_static_execution_plan("no-static-replay", &operation_registry);
    assert!(plan.payload().memory().static_allocations().is_empty());
    let resolved = no_static_resolved_model_plan(&plan, "no-static-replay", &operation_registry);
    let (resources, runtime) = provision_no_static_plan_runtime(&plan);
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &resources,
        &runtime,
        &resolved,
        &operation_registry,
        "run.no-static-replay",
        "request.no-static-replay",
        1,
    );
    assert!(active.static_pool_id().is_none());
    assert!(active.static_provisioning_identity().is_none());
    assert!(active.static_entries().is_empty());
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 1);

    let missing_cleanup = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Pending,
        &completions,
        &[],
        &[],
    );
    assert!(ReplayIdentity::from_evidence(&missing_cleanup).is_err());

    let pending = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &completions,
        &[],
        &[],
    );
    let pending_identity = ReplayIdentity::from_evidence(&pending).unwrap();
    assert_eq!(
        pending_identity.cleanup_status(),
        ReplayCleanupStatus::CleanupPending
    );
    assert!(pending_identity.pool_journal_fingerprint().is_none());
    assert!(pending_identity.plan_cleanup_fingerprint().is_none());

    let close_receipt = close_plan_runtime(resources);
    assert_eq!(close_receipt.released_static_resources(), 0);
    assert!(close_receipt.evidence().static_pool_identity().is_none());
    let clean = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
    );
    let clean_identity = ReplayIdentity::from_evidence(&clean).unwrap();
    assert_eq!(
        clean_identity.cleanup_status(),
        ReplayCleanupStatus::Completed
    );
    assert!(clean_identity.pool_journal_fingerprint().is_none());
    assert!(clean_identity.plan_cleanup_fingerprint().is_some());
    assert_eq!(
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&clean_identity).unwrap())
            .unwrap()
            .revalidate(&clean)
            .unwrap(),
        clean_identity
    );
    let mut tampered_cleanup = serde_json::to_value(&clean_identity).unwrap();
    tampered_cleanup["plan_cleanup_fingerprint"] = json!(sha('0'));
    assert!(
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&tampered_cleanup).unwrap())
            .unwrap()
            .revalidate(&clean)
            .is_err()
    );

    let (foreign_resources, _) = provision_no_static_plan_runtime(&plan);
    let foreign_close = close_plan_runtime(foreign_resources);
    let foreign_cleanup = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&foreign_close),
        &completions,
        &[],
        &[],
    );
    assert!(ReplayIdentity::from_evidence(&foreign_cleanup).is_err());
}
