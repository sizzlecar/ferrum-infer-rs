use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

const EXPECTED_LEGACY_AUTHORITY_CASES: usize = 13;
const EXPECTED_CANCEL_DISPATCH_CASES: usize = 16;
const EXPECTED_COMPLETION_CASES: usize = 200;
const EXPECTED_CASES: usize = 299;
const COMPLETION_DROP_TEST_WORKERS: usize = 1;
const MAX_COMPLETION_DROP_TEST_WORKERS: usize = 2;
const _: () = assert!(
    COMPLETION_DROP_TEST_WORKERS == 1
        && COMPLETION_DROP_TEST_WORKERS <= MAX_COMPLETION_DROP_TEST_WORKERS
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

fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}

fn suppress_expected_panic_hook<T>(action: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action));
    std::panic::set_hook(previous);
    match outcome {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
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
        FAMILY.get_or_init(|| id("family.device-operation-contract"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.device-operation")])
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
        Ok(id("metadata.device-operation"))
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
                reason: "fixture requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.device-operation-composite"),
            layout_id: id("weight-layout.device-operation-composite"),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: id("weight.component.left"),
                    role: WeightComponentRole::Values,
                    external_names: vec!["weight.left.bin".to_owned()],
                    dimensions: vec![2],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
                WeightComponentSpec {
                    id: id("weight.component.right"),
                    role: WeightComponentRole::Values,
                    external_names: vec!["weight.right.bin".to_owned()],
                    dimensions: vec![2],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![4],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Composite {
                    parts: vec![
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: id("weight.component.left"),
                            }),
                            logical_offsets: vec![0],
                            extents: vec![2],
                        },
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: id("weight.component.right"),
                            }),
                            logical_offsets: vec![2],
                            extents: vec![2],
                        },
                    ],
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
                    inputs: vec![id("value.input"), id("value.weight")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            Vec::new(),
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
    let device_id: DeviceId = id("device.device-operation.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.device-operation"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        ContractVersion::new(1, 0),
        device_id.clone(),
        capabilities.clone(),
        BTreeSet::from([id("weight-format.device-operation-composite")]),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation),
        "resource-estimator.device-operation",
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
            id("provider.engine.device-operation"),
            ContractVersion::new(1, 0),
            sha('e'),
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
                reason: "test operation signature mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderBehavior {
    Success,
    WrongIdentity,
    WrongPhase,
}

#[derive(Default)]
struct ProviderTrace {
    encode_calls: u64,
    last_participant_count: usize,
    last_work_sequences: u32,
    component_resources: BTreeSet<ResourceId>,
    view_resources: BTreeSet<ResourceId>,
}

struct TestProvider {
    descriptor: OperationProviderDescriptor,
    behavior: Arc<Mutex<ProviderBehavior>>,
    trace: Arc<Mutex<ProviderTrace>>,
}

impl OperationResourceEstimator for TestProvider {
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

impl OperationProvider<TestRuntime> for TestProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<TestCommand, OperationFailure> {
        let mut trace = self.trace.lock().unwrap();
        trace.encode_calls += 1;
        trace.last_participant_count = invocation.participants().len();
        trace.last_work_sequences = invocation.work_shape().immediate_sequences();
        let participant = &invocation.participants()[0];
        trace.component_resources = participant
            .bindings()
            .iter()
            .find(|binding| binding.value_id().as_str() == "value.weight")
            .unwrap()
            .storage()
            .components()
            .iter()
            .map(|component| component.resource_id().clone())
            .collect();
        trace.view_resources = participant
            .views()
            .iter()
            .map(|view| view.resource_id().clone())
            .collect();
        drop(trace);
        match *self.behavior.lock().unwrap() {
            ProviderBehavior::Success => Ok(TestCommand),
            ProviderBehavior::WrongIdentity => {
                let mut parts = participant.identity().parts().clone();
                parts.request_id = id("request.provider.wrong");
                let identity = ExecutionIdentityEnvelope::new(parts).unwrap();
                Err(OperationFailure::new(
                    identity,
                    ProfilePhase::Decode,
                    "provider_failure",
                    "injected provider failure",
                    false,
                )
                .unwrap())
            }
            ProviderBehavior::WrongPhase => Err(OperationFailure::new(
                participant.identity().clone(),
                ProfilePhase::Prefill,
                "provider_failure",
                "injected provider failure",
                false,
            )
            .unwrap()),
        }
    }
}

fn policy() -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.device-operation",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 65_536,
            reserve_bytes: 128,
            maximum_active_sequences: 64,
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

fn single_binding(
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

fn node_values() -> Vec<ResolvedValueBinding> {
    vec![
        single_binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            "resource.input",
        ),
        ResolvedValueBinding::new(
            id("value.weight"),
            ResolvedValueRole::Input,
            1,
            resolved_tensor(),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Weights,
            ResolvedValueStorage::composite(vec![
                ResolvedStorageComponent::new(
                    Some(id("weight.component.left")),
                    id("resource.weight.left"),
                    0,
                    8,
                    ElementType::F32,
                )
                .unwrap(),
                ResolvedStorageComponent::new(
                    Some(id("weight.component.right")),
                    id("resource.weight.right"),
                    0,
                    8,
                    ElementType::F32,
                )
                .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap(),
        single_binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            "resource.output",
        ),
    ]
}

#[derive(Debug)]
struct TestBuffer {
    descriptor: BufferDescriptor,
}

#[derive(Debug, Default)]
struct TestStream;

#[derive(Debug)]
struct TestCommand;

#[derive(Debug, Clone, Copy)]
struct TestFence(u64);

#[derive(Debug, Clone)]
struct TestRuntimeError(&'static str);

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum SubmitBehavior {
    #[default]
    Success,
    DefinitelyNotSubmitted,
    Panic,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum FenceBehavior {
    Pending,
    #[default]
    Succeeded,
    FailedButQuiescent,
    Indeterminate,
    Panic,
}

#[derive(Default)]
struct RuntimeTrace {
    allocation_calls: u64,
    submit_calls: u64,
    synchronize_calls: u64,
    wait_fence_calls: u64,
    tamper_buffer_descriptor: bool,
    drift_on_submit: bool,
    next_fence: u64,
    submit_behavior: SubmitBehavior,
    fence_behavior: FenceBehavior,
    fence_behaviors: BTreeMap<u64, FenceBehavior>,
    wait_fence_block: Option<(Arc<Barrier>, Arc<Barrier>)>,
    synchronize_fails: bool,
    stream_failed: bool,
    describe_error_panics: bool,
}

struct TestRuntime {
    descriptor: DeviceDescriptor,
    alternate_descriptor: DeviceDescriptor,
    use_alternate_descriptor: AtomicBool,
    descriptor_reads_until_drift: AtomicU64,
    trace: Arc<Mutex<RuntimeTrace>>,
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = TestCommand;
    type Fence = TestFence;
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        if self
            .descriptor_reads_until_drift
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok_and(|remaining| remaining == 1)
        {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        if self.use_alternate_descriptor.load(Ordering::Acquire) {
            &self.alternate_descriptor
        } else {
            &self.descriptor
        }
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        self.trace.lock().unwrap().allocation_calls += 1;
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
        let mut descriptor = buffer.descriptor.clone();
        if self.trace.lock().unwrap().tamper_buffer_descriptor {
            descriptor.size_bytes += 1;
        }
        descriptor
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(TestStream)
    }

    fn stream_state(&self, _stream: &Self::Stream) -> StreamState {
        if self.trace.lock().unwrap().stream_failed {
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
        Ok(TestCommand)
    }

    fn encode_upload(
        &self,
        _source: &[u8],
        _source_layout: HostTransferLayout,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand)
    }

    fn encode_zero(
        &self,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand)
    }

    fn submit(
        &self,
        _stream: &mut Self::Stream,
        _command: Self::Command,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        let (drift, behavior, fence) = {
            let mut trace = self.trace.lock().unwrap();
            trace.submit_calls += 1;
            trace.next_fence += 1;
            (
                trace.drift_on_submit,
                trace.submit_behavior,
                TestFence(trace.next_fence),
            )
        };
        match behavior {
            SubmitBehavior::DefinitelyNotSubmitted => {
                return Err(DefinitelyNotSubmitted::new(TestRuntimeError(
                    "definitely-not-submitted",
                )));
            }
            SubmitBehavior::Panic => panic!("injected submit panic"),
            SubmitBehavior::Success => {}
        }
        if drift {
            self.use_alternate_descriptor.store(true, Ordering::Release);
        }
        Ok(fence)
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        assert!(fence.0 > 0);
        let trace = self.trace.lock().unwrap();
        let behavior = trace
            .fence_behaviors
            .get(&fence.0)
            .copied()
            .unwrap_or(trace.fence_behavior);
        drop(trace);
        match behavior {
            FenceBehavior::Pending => FenceQuery::Pending,
            FenceBehavior::Succeeded => FenceQuery::Terminal(DeviceTerminal::Succeeded),
            FenceBehavior::FailedButQuiescent => FenceQuery::Terminal(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError("terminal-failure")),
            ),
            FenceBehavior::Indeterminate => {
                FenceQuery::Indeterminate(TestRuntimeError("fence-indeterminate"))
            }
            FenceBehavior::Panic => panic!("injected query panic"),
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminal<Self::Error>, FenceIndeterminate<Self::Error>> {
        assert!(fence.0 > 0);
        let (behavior, block) = {
            let mut trace = self.trace.lock().unwrap();
            trace.wait_fence_calls += 1;
            let behavior = trace
                .fence_behaviors
                .get(&fence.0)
                .copied()
                .unwrap_or(trace.fence_behavior);
            (behavior, trace.wait_fence_block.take())
        };
        if let Some((entered, release)) = block {
            entered.wait();
            release.wait();
        }
        match behavior {
            FenceBehavior::Succeeded => Ok(DeviceTerminal::Succeeded),
            FenceBehavior::FailedButQuiescent => Ok(DeviceTerminal::FailedButQuiescent(
                TestRuntimeError("terminal-failure"),
            )),
            FenceBehavior::Pending | FenceBehavior::Indeterminate => Err(FenceIndeterminate::new(
                TestRuntimeError("fence-indeterminate"),
            )),
            FenceBehavior::Panic => panic!("injected wait panic"),
        }
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), Self::Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.synchronize_calls += 1;
        if trace.synchronize_fails {
            Err(TestRuntimeError("synchronize-failed"))
        } else {
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
        assert!(
            !self.trace.lock().unwrap().describe_error_panics,
            "injected describe_error panic"
        );
        DeviceErrorReport::new("test_runtime", error.to_string(), false)
    }
}

#[derive(Default)]
struct DriverTrace {
    calls: u64,
}

struct TestDriver {
    runtime: Arc<TestRuntime>,
    trace: Arc<Mutex<DriverTrace>>,
}

impl fmt::Debug for TestDriver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TestDriver")
            .field("device", &self.runtime.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl ResourceTransactionDriver for TestDriver {
    type Buffer = TestBuffer;
    type Runtime = TestRuntime;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.runtime.descriptor.id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime.descriptor.runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.runtime.descriptor.total_memory_bytes
    }

    fn reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        Ok(())
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        context
            .allocate(&request)
            .map_err(|_| resource_failure("allocation"))
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
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
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
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _expected: &ResourceReservation,
        _actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        drop(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        drop(ownership);
    }
}

fn resource_failure(code: &str) -> ResourceDriverFailure {
    ResourceDriverFailure::new(
        FailureEnvelope::new(FailureDomain::Resource, code, "resource failure", false).unwrap(),
    )
    .unwrap()
}

fn runtime(catalog: &CapabilityCatalog) -> (Arc<TestRuntime>, Arc<Mutex<RuntimeTrace>>) {
    let trace = Arc::new(Mutex::new(RuntimeTrace::default()));
    let descriptor = catalog.device().clone();
    let mut alternate_descriptor = descriptor.clone();
    alternate_descriptor.runtime_implementation_fingerprint = sha('f');
    (
        Arc::new(TestRuntime {
            descriptor,
            alternate_descriptor,
            use_alternate_descriptor: AtomicBool::new(false),
            descriptor_reads_until_drift: AtomicU64::new(0),
            trace: Arc::clone(&trace),
        }),
        trace,
    )
}

struct TestModelRegistry {
    registration: TypedFamilyRegistration<TestFamily>,
}

impl TestModelRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }
}

impl ModelFamilyRegistry for TestModelRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
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

fn operation_registry(
    catalog: &CapabilityCatalog,
    behavior: Arc<Mutex<ProviderBehavior>>,
    trace: Arc<Mutex<ProviderTrace>>,
) -> OperationRuntimeRegistry<TestRuntime> {
    OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: catalog.operation(&id("operation.main")).unwrap().clone(),
        })],
        vec![Box::new(TestProvider {
            descriptor: catalog.providers_for(&id("operation.main")).unwrap()[0].clone(),
            behavior,
            trace,
        })],
    )
    .unwrap()
}

fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> PlanNodeResolution {
    PlanNodeResolution::resolve(
        family,
        catalog,
        runtime_policy,
        &registry.planning(),
        id("node.main"),
        node_values(),
        BTreeSet::new(),
        None,
    )
    .unwrap()
}

fn resolved_model_plan(
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> (ResolvedModelPlan, ExecutionPlan) {
    let model_registry = TestModelRegistry::new();
    let family = model_registry
        .registration
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let resolutions = vec![node_resolution(
        &family,
        &catalog,
        &runtime_policy,
        registry,
    )];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &runtime_policy, resolutions.clone()).unwrap(),
    )
    .unwrap();
    let config_fingerprint = family.config_fingerprint().to_owned();
    let inputs = ResolvedModelPlanInputs {
        original_source: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/device-operation-model".to_owned(),
            requested_revision: Some("main".to_owned()),
        },
        resolved_source: ResolvedModelSource {
            canonical_location: "repo/device-operation-model".to_owned(),
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
        external_metadata_id: id("metadata.device-operation"),
        prepared_family: family,
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.device-operation"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1024,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime_policy.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.device-operation"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('e'),
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
        let artifact_id: ResolutionArtifactId = id(format!("artifact.device-operation.{index}"));
        let path = "/chosen".to_owned();
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "fixture.device-operation".to_owned(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e'))
                        .unwrap(),
                    revision: "fixture-v1".to_owned(),
                    artifact_locator: format!("device-operation/{index}"),
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
                id(format!("reason.device-operation.{index}")),
                artifact_id,
                path,
            )
            .unwrap(),
        );
    }
    let context = ResolvedPlanValidationContext::new(
        &model_registry,
        &evidence,
        &resolutions,
        catalog.device(),
        &catalog,
        &runtime_policy,
    );
    (
        ResolvedModelPlan::new(inputs, bindings, &context).unwrap(),
        plan,
    )
}

fn plan_for_registry(registry: &OperationRuntimeRegistry<TestRuntime>) -> ExecutionPlan {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    ExecutionPlan::build(
        PlanBuildRequest::new(
            &family,
            &catalog,
            &runtime_policy,
            vec![node_resolution(
                &family,
                &catalog,
                &runtime_policy,
                registry,
            )],
        )
        .unwrap(),
    )
    .unwrap()
}

fn plan_runtime_resources(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(&runtime),
            id("request.device-operation.provision"),
        )
        .unwrap()
        .into_parts();
    let admission = match provisioning {
        StaticProvisioning::Required(admission) => admission,
        StaticProvisioning::NoStatic(_) => {
            panic!("device operation fixture requires static/backing provisioning")
        }
    };
    let identity = ResourceTransactionIdentity::for_admission(
        admission.binding(),
        id("run.device-operation.provision"),
        id("transaction.device-operation.provision"),
    );
    let driver = TestDriver {
        runtime,
        trace: Arc::new(Mutex::new(DriverTrace::default())),
    };
    let committed = ResourceTransaction::begin(driver, identity, admission)
        .unwrap()
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
                    request_maintenance_attempts < 3,
                    "request admission did not converge after bounded maintenance"
                );
                request_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            RequestResourceAdmissionDecision::Deferred(_) => {
                panic!("device operation fixture request logical admission deferred")
            }
            RequestResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("device operation fixture request admission rejected")
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
                    sequence_maintenance_attempts < 3,
                    "sequence admission did not converge after bounded maintenance"
                );
                sequence_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("device operation fixture sequence logical admission deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("device operation fixture sequence admission rejected")
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
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => {
                assert_eq!(step.work_shape().participants().len(), 1);
                assert_eq!(step.work_shape().immediate_sequences(), 1);
                assert_eq!(step.work_shape().immediate_tokens(), 1);
                assert_eq!(step.work_shape().immediate_pages(), 0);
                assert_eq!(step.work_shape().fit_sequences(), 1);
                assert_eq!(step.work_shape().fit_tokens(), 1);
                assert_eq!(step.work_shape().fit_pages(), 0);
                assert_eq!(step.work_shape().fingerprint().len(), 64);
                assert_eq!(step.claimed_backing().fingerprint().len(), 64);
                match step.claimed_backing().logical_capacity() {
                    Some(capacity) => assert_eq!(
                        step.claimed_backing().demand().immediate_claim(),
                        capacity.claims()
                    ),
                    None => assert!(step.claimed_backing().demand().immediate_claim().is_empty()),
                }
                return step;
            }
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
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
    for attempt in 0..=3 {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => {
                assert_eq!(invocation.work_shape(), step.work_shape());
                assert_eq!(invocation.claimed_backing().fingerprint().len(), 64);
                assert_eq!(invocation.work_shape().fingerprint().len(), 64);
                return invocation;
            }
            InvocationResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            InvocationResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("invocation backing did not converge after bounded maintenance")
            }
            InvocationResourceAdmissionDecision::Deferred(_) => {
                panic!("single-participant invocation unexpectedly deferred")
            }
            InvocationResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("single-participant invocation unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded invocation admission loop always returns or panics")
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

struct Fixture {
    registry: OperationRuntimeRegistry<TestRuntime>,
    impostor_registry: OperationRuntimeRegistry<TestRuntime>,
    resolved: ResolvedModelPlan,
    plan: ExecutionPlan,
    impostor_plan_hash: PlanHash,
    runtime: Arc<TestRuntime>,
    runtime_trace: Arc<Mutex<RuntimeTrace>>,
    provider_behavior: Arc<Mutex<ProviderBehavior>>,
    provider_trace: Arc<Mutex<ProviderTrace>>,
    plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
}

fn fixture() -> Fixture {
    let catalog = catalog();
    let provider_behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let provider_trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = operation_registry(
        &catalog,
        Arc::clone(&provider_behavior),
        Arc::clone(&provider_trace),
    );
    let (resolved, plan) = resolved_model_plan(&registry);
    let impostor_registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::WrongPhase)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let impostor_plan_hash = plan_for_registry(&impostor_registry).plan_hash().clone();
    let (runtime, runtime_trace) = runtime(&catalog);
    let plan_resources = plan_runtime_resources(&plan, Arc::clone(&runtime));
    Fixture {
        registry,
        impostor_registry,
        resolved,
        plan,
        impostor_plan_hash,
        runtime,
        runtime_trace,
        provider_behavior,
        provider_trace,
        plan_resources,
    }
}

fn close_plan_runtime(plan_resources: Arc<PlanRuntimeResources<TestRuntime>>, passed: &mut usize) {
    match PlanRuntimeResources::close(plan_resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            check(passed, receipt.released_static_resources() == 2)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
}

fn device_identity_parts(
    run_id: &str,
    request_id: &str,
    device_id: DeviceId,
    runtime_fingerprint: String,
) -> ExecutionIdentityParts {
    ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: id(run_id),
        request_id: id(request_id),
        sequence: 1,
        plan_id: None,
        plan_hash: None,
        frame_id: None,
        node_invocation_id: None,
        node_id: None,
        operation_id: None,
        provider_id: None,
        device_id: Some(device_id),
        resource_pool_id: None,
        resource_pool_identity_fingerprint: None,
        provisioning_run_id: None,
        provisioning_request_id: None,
        transaction_id: None,
        active_sequence_slot: None,
        admission_generation: None,
        activation_epoch: None,
        runtime_implementation_fingerprint: Some(runtime_fingerprint),
        active_sequence_fingerprint: None,
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id("span.device-operation"),
        parent_span_id: None,
        async_links: Vec::new(),
    }
}

fn operation_identity(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    frame_id: ExecutionFrameId,
    invocation_id: NodeInvocationId,
) -> ExecutionIdentityEnvelope {
    let node = &plan.payload().nodes()[0];
    let provisioning = active.static_provisioning_identity();
    ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: active.run_id().clone(),
        request_id: active.request_id().clone(),
        sequence: 1,
        plan_id: Some(plan.payload().plan_id().clone()),
        plan_hash: Some(plan.plan_hash().clone()),
        frame_id: Some(frame_id),
        node_invocation_id: Some(invocation_id),
        node_id: Some(node.id().clone()),
        operation_id: Some(node.operation_id().clone()),
        provider_id: Some(node.selection().selected_provider().clone()),
        device_id: Some(plan.payload().device_id().clone()),
        resource_pool_id: active.static_pool_id(),
        resource_pool_identity_fingerprint: active.static_pool_identity_fingerprint(),
        provisioning_run_id: provisioning.map(|identity| identity.run_id().clone()),
        provisioning_request_id: provisioning.map(|identity| identity.request_id().clone()),
        transaction_id: provisioning.map(|identity| identity.transaction_id().clone()),
        active_sequence_slot: Some(active.sequence_authority().sparse_id()),
        admission_generation: Some(active.sequence_authority().generation()),
        activation_epoch: Some(active.activation_epoch()),
        runtime_implementation_fingerprint: Some(
            active.runtime_implementation_fingerprint().to_owned(),
        ),
        active_sequence_fingerprint: Some(active.fingerprint().to_owned()),
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id("span.device-operation.node"),
        parent_span_id: None,
        async_links: Vec::new(),
    })
    .unwrap()
}

fn revalidate_plan_for_registry(
    bytes: &[u8],
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let resolution = node_resolution(&family, &catalog, &runtime_policy, registry);
    ExecutionPlan::from_json_validated(bytes, &family, &catalog, &runtime_policy, vec![resolution])
        .unwrap()
}

fn serialization_message(error: VNextError) -> String {
    match error {
        VNextError::Serialization { message, .. } => message,
        other => panic!("expected serialization error, got {other}"),
    }
}

fn descriptor_and_registry_contract(passed: &mut usize) {
    let catalog = catalog();
    let mut zero_capacity = catalog.device().clone();
    zero_capacity.total_memory_bytes = 0;
    check(passed, zero_capacity.validate().is_err());

    let mut invalid_runtime = catalog.device().clone();
    invalid_runtime.runtime_implementation_fingerprint = "not-a-sha".to_owned();
    check(passed, invalid_runtime.validate().is_err());

    let behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let descriptor = catalog.providers_for(&id("operation.main")).unwrap()[0].clone();
    let duplicate = OperationRuntimeRegistry::<TestRuntime>::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
        })],
        vec![
            Box::new(TestProvider {
                descriptor: descriptor.clone(),
                behavior: Arc::clone(&behavior),
                trace: Arc::clone(&trace),
            }),
            Box::new(TestProvider {
                descriptor,
                behavior,
                trace,
            }),
        ],
    );
    check(passed, duplicate.is_err());
}

fn device_failure_contract(
    runtime: &TestRuntime,
    plan: &ExecutionPlan,
    operation_identity: &ExecutionIdentityEnvelope,
    passed: &mut usize,
) {
    let descriptor = runtime.descriptor();
    let device_only = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-only",
        "request.device-only",
        descriptor.id.clone(),
        descriptor.runtime_implementation_fingerprint.clone(),
    ))
    .unwrap();
    let failure =
        classify_device_error(runtime, device_only, &TestRuntimeError("device-only")).unwrap();
    check(passed, failure.failure().domain() == FailureDomain::Device);

    let missing_runtime = {
        let mut parts = device_identity_parts(
            "run.device-missing-runtime",
            "request.device-missing-runtime",
            descriptor.id.clone(),
            descriptor.runtime_implementation_fingerprint.clone(),
        );
        parts.runtime_implementation_fingerprint = None;
        ExecutionIdentityEnvelope::new(parts)
    };
    check(passed, missing_runtime.is_err());

    let wrong_runtime = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-wrong-runtime",
        "request.device-wrong-runtime",
        descriptor.id.clone(),
        sha('f'),
    ))
    .unwrap();
    check(
        passed,
        classify_device_error(runtime, wrong_runtime, &TestRuntimeError("wrong-runtime")).is_err(),
    );

    let wrong_device = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-wrong-device",
        "request.device-wrong-device",
        id("device.other"),
        descriptor.runtime_implementation_fingerprint.clone(),
    ))
    .unwrap();
    check(
        passed,
        classify_device_error(runtime, wrong_device, &TestRuntimeError("wrong-device")).is_err(),
    );

    let mut plan_parts = device_identity_parts(
        "run.device-plan",
        "request.device-plan",
        descriptor.id.clone(),
        descriptor.runtime_implementation_fingerprint.clone(),
    );
    plan_parts.plan_id = Some(plan.payload().plan_id().clone());
    plan_parts.plan_hash = Some(plan.plan_hash().clone());
    let plan_identity = ExecutionIdentityEnvelope::new(plan_parts).unwrap();
    check(
        passed,
        classify_device_error(runtime, plan_identity, &TestRuntimeError("plan")).is_ok(),
    );
    check(
        passed,
        classify_device_error(
            runtime,
            operation_identity.clone(),
            &TestRuntimeError("active-operation"),
        )
        .is_ok(),
    );

    let mut resource_parts = operation_identity.parts().clone();
    resource_parts.frame_id = None;
    resource_parts.node_invocation_id = None;
    resource_parts.node_id = None;
    resource_parts.operation_id = None;
    resource_parts.provider_id = None;
    resource_parts.active_sequence_slot = None;
    resource_parts.admission_generation = None;
    resource_parts.activation_epoch = None;
    resource_parts.active_sequence_fingerprint = None;
    resource_parts.resource_id = Some(id("resource.input"));
    resource_parts.resource_generation = Some(1);
    let resource_identity = ExecutionIdentityEnvelope::new(resource_parts).unwrap();
    check(
        passed,
        classify_device_error(runtime, resource_identity, &TestRuntimeError("resource")).is_ok(),
    );
}

fn operation_dispatch_contract(fixture: Fixture, passed: &mut usize) {
    let Fixture {
        registry,
        impostor_registry,
        resolved,
        plan,
        impostor_plan_hash,
        runtime,
        runtime_trace,
        provider_behavior,
        provider_trace,
        plan_resources,
    } = fixture;
    let node = &plan.payload().nodes()[0];
    check(passed, plan.plan_hash() == &impostor_plan_hash);
    let plan_bytes = plan.to_json().unwrap();
    let revalidated = revalidate_plan_for_registry(&plan_bytes, &registry);
    check(passed, revalidated == plan);
    let revalidated_impostor = revalidate_plan_for_registry(&plan_bytes, &impostor_registry);
    check(
        passed,
        revalidated_impostor.plan_hash() == plan.plan_hash()
            && revalidated_impostor.to_json().unwrap() == plan.to_json().unwrap()
            && revalidated_impostor != plan,
    );
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let impostor_resolution =
        node_resolution(&family, &catalog, &runtime_policy, &impostor_registry);
    check(
        passed,
        plan.validate_against(&family, &catalog, &runtime_policy, &[impostor_resolution])
            .is_err(),
    );
    check(passed, registry.bind(&resolved, node.id()).is_ok());
    check(
        passed,
        impostor_registry.bind(&resolved, node.id()).is_err(),
    );
    let provider = registry.bind(&resolved, node.id()).unwrap();

    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.execute",
        "request.device-operation.execute",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    device_failure_contract(&runtime, &plan, &identity, passed);
    assert!(step.try_retire_normal().is_ok());

    type IdentityMutation = fn(&mut ExecutionIdentityParts);
    let mutations: [IdentityMutation; 13] = [
        |parts| parts.plan_id = Some(id(format!("plan/sha256/{}", sha('f')))),
        |parts| parts.plan_hash = Some(serde_json::from_value(json!(sha('f'))).unwrap()),
        |parts| {
            parts.resource_pool_id =
                Some(ResourcePoolId::try_from(parts.resource_pool_id.unwrap().get() + 1).unwrap())
        },
        |parts| parts.resource_pool_identity_fingerprint = Some(sha('f')),
        |parts| parts.activation_epoch = Some(parts.activation_epoch.unwrap() + 1),
        |parts| parts.active_sequence_slot = Some(parts.active_sequence_slot.unwrap() + 1),
        |parts| parts.request_id = id("request.device-operation.wrong"),
        |parts| parts.run_id = id("run.device-operation.wrong"),
        |parts| {
            parts.frame_id =
                Some(ExecutionFrameId::try_from(parts.frame_id.unwrap().get() + 1).unwrap())
        },
        |parts| {
            parts.node_invocation_id = Some(
                NodeInvocationId::try_from(parts.node_invocation_id.unwrap().get() + 1).unwrap(),
            )
        },
        |parts| parts.provider_id = Some(id("provider.operation.wrong")),
        |parts| parts.node_id = Some(id("node.wrong")),
        |parts| parts.runtime_implementation_fingerprint = Some(sha('f')),
    ];

    for mutate in mutations {
        let step = begin_single_participant_step(&plan_resources, &batch);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
        let mut parts = operation_identity(&plan, &active, frame_id, invocation_id)
            .parts()
            .clone();
        mutate(&mut parts);
        let wrong = ExecutionIdentityEnvelope::new(parts).unwrap();
        check(
            passed,
            matches!(
                encode_and_submit_single(
                    &provider,
                    &resolved,
                    &wrong,
                    &frame_id,
                    &invocation_id,
                    node.id(),
                    &active,
                    admit_single_participant_invocation(&plan_resources, &step, node.id()),
                    &lane,
                    &reaper,
                ),
                Err(OperationDispatchError::Contract(_))
            ),
        );
        assert!(step.try_retire_normal().is_ok());
    }
    for completed in [true, false] {
        let step = begin_single_participant_step(&plan_resources, &batch);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
        let mut parts = operation_identity(&plan, &active, frame_id, invocation_id)
            .parts()
            .clone();
        if completed {
            parts.completed_sequence_fingerprint = Some(sha('c'));
        } else {
            parts.aborted_sequence_fingerprint = Some(sha('a'));
        }
        let terminal = ExecutionIdentityEnvelope::new(parts).unwrap();
        let failure_rejected = OperationFailure::new(
            terminal.clone(),
            ProfilePhase::Decode,
            "operation_failed",
            "terminal sequence cannot report an active operation failure",
            false,
        )
        .is_err();
        let dispatch_rejected = matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &terminal,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        );
        check(passed, failure_rejected && dispatch_rejected);
        assert!(step.try_retire_normal().is_ok());
    }
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);

    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let mut resource_item = operation_identity(&plan, &active, frame_id, invocation_id)
        .parts()
        .clone();
    resource_item.resource_id = Some(id("resource.input"));
    resource_item.resource_generation = Some(active.sequence_authority().generation());
    let resource_item = ExecutionIdentityEnvelope::new(resource_item).unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &resource_item,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    assert!(step.try_retire_normal().is_ok());

    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let mut resource_batch = operation_identity(&plan, &active, frame_id, invocation_id)
        .parts()
        .clone();
    resource_batch.resource_batch_fingerprint = Some(sha('f'));
    let resource_batch = ExecutionIdentityEnvelope::new(resource_batch).unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &resource_batch,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    assert!(step.try_retire_normal().is_ok());
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);

    *provider_behavior.lock().unwrap() = ProviderBehavior::WrongIdentity;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    assert!(step.try_retire_normal().is_ok());

    *provider_behavior.lock().unwrap() = ProviderBehavior::WrongPhase;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    assert!(step.try_retire_normal().is_ok());

    *provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let receipt = encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    check(
        passed,
        receipt.batch_identity().participants()[0].identity() == &identity,
    );
    check(
        passed,
        matches!(receipt.poll(), Ok(CompletionObservation::Terminal(_))),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 3);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(
        passed,
        provider_trace.lock().unwrap().component_resources
            == BTreeSet::from([id("resource.weight.left"), id("resource.weight.right")]),
    );
    check(
        passed,
        provider_trace.lock().unwrap().view_resources
            == BTreeSet::from([
                id("resource.input"),
                id("resource.output"),
                id("resource.weight.left"),
                id("resource.weight.right"),
            ]),
    );
    let step_receipt = step.try_retire_normal().unwrap();
    check(
        passed,
        step_receipt.participants()[0]
            .assignment()
            .sequence_authority()
            == active.sequence_authority(),
    );
    drop(receipt);

    let encode_before_tamper = provider_trace.lock().unwrap().encode_calls;
    let submit_before_tamper = runtime_trace.lock().unwrap().submit_calls;
    runtime_trace.lock().unwrap().tamper_buffer_descriptor = true;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(
        passed,
        provider_trace.lock().unwrap().encode_calls == encode_before_tamper,
    );
    check(
        passed,
        runtime_trace.lock().unwrap().submit_calls == submit_before_tamper,
    );
    runtime_trace.lock().unwrap().tamper_buffer_descriptor = false;
    assert!(step.try_retire_normal().is_ok());

    wire_limit_contract(&plan, &identity, passed);
    check(passed, session.try_complete().is_ok());
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    close_plan_runtime(plan_resources, passed);
}

fn cancel_dispatch_linearization_contract(initial_fixture: Fixture, passed: &mut usize) {
    let start = *passed;
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = initial_fixture;
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.cancel-before-dispatch",
        "request.cancel-before-dispatch",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let invocation = admit_single_participant_invocation(&plan_resources, &step, node.id());
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();

    let cancel = session.request_cancel().unwrap();
    check(
        passed,
        cancel.active_frame() == Some(frame_id) && cancel.participant_flights() == 1,
    );
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                invocation,
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    check(
        passed,
        reaper.retained_count() == 0 && lane.in_flight_count() == 0,
    );
    let retired = step.try_retire_normal().unwrap();
    check(
        passed,
        retired.participants()[0].disposition()
            == StepParticipantRetirementDisposition::DiscardedCancelled,
    );
    let terminal = session.try_abort().unwrap();
    check(
        passed,
        terminal.disposition() == SequenceSessionTerminalDisposition::Aborted,
    );
    drop(terminal);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);

    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture();
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.dispatch-before-cancel",
        "request.dispatch-before-cancel",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let completion = encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    )
    .unwrap();

    let cancel = session.request_cancel().unwrap();
    check(
        passed,
        cancel.active_frame() == Some(frame_id) && cancel.participant_flights() == 1,
    );
    check(
        passed,
        matches!(completion.poll(), Ok(CompletionObservation::Terminal(_))),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 1);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(
        passed,
        reaper.retained_count() == 0 && lane.in_flight_count() == 0,
    );
    let retired = step.try_retire_normal().unwrap();
    check(
        passed,
        retired.participants()[0].disposition()
            == StepParticipantRetirementDisposition::DiscardedCancelled,
    );
    let terminal = session.try_abort().unwrap();
    check(
        passed,
        terminal.disposition() == SequenceSessionTerminalDisposition::Aborted,
    );
    drop(terminal);
    drop(completion);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);
    assert_eq!(*passed - start, EXPECTED_CANCEL_DISPATCH_CASES);
}

fn wire_limit_contract(
    plan: &ExecutionPlan,
    identity: &ExecutionIdentityEnvelope,
    passed: &mut usize,
) {
    let plan_bytes = plan.to_json().unwrap();
    check(passed, ExecutionPlan::decode_untrusted(&plan_bytes).is_ok());
    let at_plan_limit = vec![b' '; MAX_EXECUTION_PLAN_WIRE_BYTES];
    let message =
        serialization_message(ExecutionPlan::decode_untrusted(&at_plan_limit).unwrap_err());
    check(passed, !message.contains("exceeds limit"));
    let over_plan_limit = vec![b' '; MAX_EXECUTION_PLAN_WIRE_BYTES + 1];
    let message =
        serialization_message(ExecutionPlan::decode_untrusted(&over_plan_limit).unwrap_err());
    check(passed, message.contains("exceeds limit"));

    let failure = OperationFailure::new(
        identity.clone(),
        ProfilePhase::Decode,
        "operation_failed",
        "operation failed",
        false,
    )
    .unwrap();
    let failure_bytes = serde_json::to_vec(&failure).unwrap();
    let unvalidated = OperationFailure::decode_untrusted(&failure_bytes).unwrap();
    check(
        passed,
        unvalidated
            .revalidate(identity, ProfilePhase::Decode)
            .is_ok(),
    );
    let at_failure_limit = vec![b' '; MAX_OPERATION_FAILURE_WIRE_BYTES];
    let message =
        serialization_message(OperationFailure::decode_untrusted(&at_failure_limit).unwrap_err());
    check(passed, !message.contains("exceeds limit"));
    let over_failure_limit = vec![b' '; MAX_OPERATION_FAILURE_WIRE_BYTES + 1];
    let message =
        serialization_message(OperationFailure::decode_untrusted(&over_failure_limit).unwrap_err());
    check(passed, message.contains("exceeds limit"));
}

fn submit_descriptor_drift_contract(fixture: Fixture, passed: &mut usize) {
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture;
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.submit-drift",
        "request.device-operation.submit-drift",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    runtime_trace.lock().unwrap().drift_on_submit = true;
    let completion = match encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    ) {
        Err(OperationDispatchError::PostSubmitContract { completion, .. }) => completion,
        _ => panic!("submit descriptor drift did not retain a completion handle"),
    };
    check(passed, reaper.retained_count() == 1);
    check(passed, provider_trace.lock().unwrap().encode_calls == 1);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(passed, lane.is_fail_closed());
    let step = match step.try_retire_normal() {
        Err(failure) => failure.into_step(),
        Ok(_) => panic!("pending drift fence released invocation resources"),
    };
    let terminal = match completion.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("descriptor drift did not produce a quiescent terminal: {other:?}"),
    };
    check(
        passed,
        matches!(
            terminal.disposition(),
            OperationCompletionDisposition::ContractFailedButQuiescent(failure)
                if failure.reason().contains("descriptor")
        ),
    );
    runtime
        .use_alternate_descriptor
        .store(false, Ordering::Release);
    runtime_trace.lock().unwrap().drift_on_submit = false;
    check(passed, reaper.retained_count() == 0);
    check(passed, lane.in_flight_count() == 0);
    check(passed, step.try_retire_normal().is_ok());
    check(passed, session.try_complete().is_ok());
    drop(completion);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    close_plan_runtime(plan_resources, passed);
}

fn legacy_source_seals_sequence_and_cannot_authorize_other_session(
    fixture: Fixture,
    passed: &mut usize,
) {
    let start = *passed;
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture;
    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.legacy-authority",
        "request.device-operation.legacy-authority",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let legacy_binding = TrustedActiveSequenceBinding::from_permit(&permit).unwrap();
    let legacy_completion = permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        legacy_completion.activation_epoch() == legacy_binding.activation_epoch()
            && legacy_completion.sequence_authority() == legacy_binding.sequence_authority(),
    );
    drop(stream);

    check(
        passed,
        matches!(
            resources.open_session(),
            Err(error)
                if error
                    .to_string()
                    .contains("permanently selected for legacy streams")
        ),
    );

    let session_resources = logical_resources(
        &plan_resources,
        "run.device-operation.session-authority",
        "request.device-operation.session-authority",
    );
    let session = session_resources.open_session().unwrap();
    check(
        passed,
        legacy_binding.sequence_authority() != session.sequence_authority(),
    );
    check(
        passed,
        legacy_binding.run_id() != session_resources.run_id(),
    );
    check(
        passed,
        legacy_binding.request_id() != session_resources.request_id(),
    );
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &legacy_binding, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &legacy_binding,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    check(passed, reaper.retained_count() == 0);
    check(passed, lane.in_flight_count() == 0);
    check(passed, step.try_retire_normal().is_ok());
    check(passed, session.try_complete().is_ok());
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(session_resources);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);
    assert_eq!(*passed - start, EXPECTED_LEGACY_AUTHORITY_CASES);
}

struct CompletionHarness {
    registry: OperationRuntimeRegistry<TestRuntime>,
    resolved: ResolvedModelPlan,
    plan: ExecutionPlan,
    runtime: Arc<TestRuntime>,
    runtime_trace: Arc<Mutex<RuntimeTrace>>,
    plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
    resources: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    active: TrustedActiveSequenceBinding,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
}

impl CompletionHarness {
    fn step(&self) -> &Arc<StepResourceLease<TestRuntime>> {
        self.step.as_ref().expect("completion harness owns a step")
    }

    fn new() -> Self {
        let Fixture {
            registry,
            resolved,
            plan,
            runtime,
            runtime_trace,
            plan_resources,
            ..
        } = fixture();
        let resources = logical_resources(
            &plan_resources,
            "run.device-operation.completion",
            "request.device-operation.completion",
        );
        let session = resources.open_session().unwrap();
        let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
        let step = begin_single_participant_step(&plan_resources, &batch);
        let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
        let reaper = CompletionReaper::new();
        Self {
            registry,
            resolved,
            plan,
            runtime,
            runtime_trace,
            plan_resources,
            resources,
            session,
            batch,
            step: Some(step),
            active,
            lane,
            reaper,
        }
    }

    fn dispatch(
        &self,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        self.dispatch_on_lane(&self.lane)
    }

    fn dispatch_on_lane(
        &self,
        lane: &Arc<ExecutionLane<TestRuntime>>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        let node = &self.plan.payload().nodes()[0];
        self.dispatch_invocation_on_lane(
            admit_single_participant_invocation(&self.plan_resources, self.step(), node.id()),
            lane,
        )
    }

    fn dispatch_invocation(
        &self,
        invocation: InvocationResourceLease<TestRuntime>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        self.dispatch_invocation_on_lane(invocation, &self.lane)
    }

    fn dispatch_invocation_on_lane(
        &self,
        invocation: InvocationResourceLease<TestRuntime>,
        lane: &Arc<ExecutionLane<TestRuntime>>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        let node = &self.plan.payload().nodes()[0];
        let provider = self.registry.bind(&self.resolved, node.id()).unwrap();
        let frame_id = self.step().participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(97).unwrap();
        let identity = operation_identity(&self.plan, &self.active, frame_id, invocation_id);
        encode_and_submit_single(
            &provider,
            &self.resolved,
            &identity,
            &frame_id,
            &invocation_id,
            node.id(),
            &self.active,
            invocation,
            lane,
            &self.reaper,
        )
    }

    fn set_submit_behavior(&self, behavior: SubmitBehavior) {
        self.runtime_trace.lock().unwrap().submit_behavior = behavior;
    }

    fn set_fence_behavior(&self, behavior: FenceBehavior) {
        self.runtime_trace.lock().unwrap().fence_behavior = behavior;
    }

    fn set_fence_behavior_for(&self, fence: u64, behavior: FenceBehavior) {
        self.runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(fence, behavior);
    }

    fn set_synchronize_fails(&self, fails: bool) {
        self.runtime_trace.lock().unwrap().synchronize_fails = fails;
    }

    fn block_next_wait(&self, entered: Arc<Barrier>, release: Arc<Barrier>) {
        self.runtime_trace.lock().unwrap().wait_fence_block = Some((entered, release));
    }

    fn drift_after_descriptor_reads(&self, reads: u64) {
        self.runtime
            .descriptor_reads_until_drift
            .store(reads, Ordering::Release);
    }

    fn set_stream_failed(&self, failed: bool) {
        self.runtime_trace.lock().unwrap().stream_failed = failed;
    }

    fn set_describe_error_panics(&self, panics: bool) {
        self.runtime_trace.lock().unwrap().describe_error_panics = panics;
    }

    fn finish(mut self, passed: &mut usize) {
        check(passed, self.reaper.retained_count() == 0);
        check(passed, self.reaper.quarantined_count() == 0);
        check(passed, self.lane.in_flight_count() == 0);
        check(
            passed,
            self.step
                .take()
                .expect("completion harness owns a final step")
                .try_retire_normal()
                .is_ok(),
        );
        check(passed, self.session.try_complete().is_ok());
        drop(self.reaper);
        drop(self.lane);
        drop(self.batch);
        drop(self.session);
        drop(self.resources);
        drop(self.runtime);
        match PlanRuntimeResources::close(self.plan_resources) {
            Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
                check(passed, receipt.released_static_resources() == 2)
            }
            Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
                panic!("completion harness retained {strong_count} root references")
            }
            Err(failure) => panic!("completion harness close failed: {:?}", failure.failure()),
        }
    }
}

fn retire_step_after_deferred_cleanup(
    mut step: Arc<StepResourceLease<TestRuntime>>,
) -> Result<StepRetirementReceipt, StepFinalizationFailure<TestRuntime>> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match step.try_retire_normal() {
            Ok(receipt) => return Ok(receipt),
            Err(failure) if Instant::now() < deadline => {
                step = failure.into_step();
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(failure) => return Err(failure),
        }
    }
}

#[test]
fn completion_reaper_drop_defers_blocking_backend_recovery() {
    let harness = CompletionHarness::new();
    harness.set_fence_behavior(FenceBehavior::Pending);
    let completion = harness.dispatch().unwrap();
    assert_eq!(harness.reaper.retained_count(), 1);
    assert_eq!(harness.lane.in_flight_count(), 1);

    let entered = Arc::new(Barrier::new(COMPLETION_DROP_TEST_WORKERS + 1));
    let release = Arc::new(Barrier::new(COMPLETION_DROP_TEST_WORKERS + 1));
    harness.block_next_wait(Arc::clone(&entered), Arc::clone(&release));
    let CompletionHarness {
        runtime,
        runtime_trace,
        plan_resources,
        resources,
        session,
        batch,
        step,
        lane,
        reaper,
        ..
    } = harness;

    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_returned = std::thread::scope(|scope| {
        let drop_worker = std::thread::Builder::new()
            .name("vnext-blocking-reaper-drop".to_owned())
            .spawn_scoped(scope, move || {
                drop(reaper);
                let _ = dropped_tx.send(());
            })
            .expect("the single bounded reaper-drop worker starts");
        let drop_returned = dropped_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        drop_worker
            .join()
            .expect("the bounded reaper-drop worker does not panic");
        drop_returned
    });
    assert!(
        drop_returned,
        "CompletionReaper::drop waited on the backend"
    );
    assert_eq!(plan_resources.deferred_cleanup_status().pending(), 1);
    assert_eq!(runtime_trace.lock().unwrap().wait_fence_calls, 0);

    let maintenance_root = Arc::clone(&plan_resources);
    let (cleanup, fail_closed_while_blocked, in_flight_while_blocked) =
        std::thread::scope(|scope| {
            let cleanup_worker = std::thread::Builder::new()
                .name("vnext-completion-cleanup-recovery".to_owned())
                .spawn_scoped(scope, move || {
                    maintenance_root.maintain_deferred_cleanups(1)
                })
                .expect("the single bounded completion cleanup worker starts");
            entered.wait();
            let fail_closed_while_blocked = lane.is_fail_closed();
            let in_flight_while_blocked = lane.in_flight_count();
            release.wait();
            let cleanup = cleanup_worker
                .join()
                .expect("the bounded completion cleanup worker does not panic")
                .expect("the bounded completion cleanup call is valid");
            (cleanup, fail_closed_while_blocked, in_flight_while_blocked)
        });

    assert!(fail_closed_while_blocked);
    assert_eq!(in_flight_while_blocked, 1);
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    let retirement = retire_step_after_deferred_cleanup(step.expect("harness owns a step"))
        .expect("deferred completion cleanup converges after backend release");
    assert_eq!(retirement.participants().len(), 1);
    assert_eq!(lane.in_flight_count(), 0);
    let trace = runtime_trace.lock().unwrap();
    assert_eq!(trace.wait_fence_calls, 1);
    assert_eq!(trace.synchronize_calls, 1);
    drop(trace);
    session.try_complete().unwrap();
    drop(completion);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));

    let quarantined = CompletionHarness::new();
    quarantined.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = quarantined.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    assert!(matches!(
        quarantined.reaper.wait_slot_for_recovery(slot_id),
        Ok(CompletionObservation::Indeterminate(_))
    ));
    quarantined.set_synchronize_fails(true);
    let quarantine = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Quarantined(receipt) => receipt,
        CompletionRecoveryOutcome::Drained(_) => panic!("failed drain released ownership"),
    };
    let CompletionHarness {
        runtime,
        runtime_trace,
        plan_resources,
        resources,
        session,
        batch,
        step,
        lane,
        reaper,
        ..
    } = quarantined;
    drop(reaper);
    assert_eq!(plan_resources.deferred_cleanup_status().pending(), 1);
    let first_cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first_cleanup.quarantined(), 1);
    assert_eq!(first_cleanup.status_after().pending(), 1);
    assert!(quarantine.is_current());
    runtime_trace.lock().unwrap().synchronize_fails = false;
    let second_cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(second_cleanup.completed(), 1);
    assert_eq!(second_cleanup.status_after().pending(), 0);
    assert!(!quarantine.is_current());
    assert_eq!(lane.in_flight_count(), 0);
    assert!(step
        .expect("harness owns a step")
        .try_retire_normal()
        .is_ok());
    assert!(session.try_complete().is_ok());
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

fn completion_reaper_owns_invocations_until_quiescent_terminal(passed: &mut usize) {
    let start = *passed;
    let wrong_runtime = CompletionHarness::new();
    let (other_runtime, _) = runtime(&catalog());
    let other_lane = ExecutionLane::create(other_runtime).unwrap();
    check(
        passed,
        matches!(
            wrong_runtime.dispatch_on_lane(&other_lane),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, wrong_runtime.reaper.retained_count() == 0);
    drop(other_lane);
    wrong_runtime.finish(passed);

    let definitely_not_submitted = CompletionHarness::new();
    definitely_not_submitted.set_submit_behavior(SubmitBehavior::DefinitelyNotSubmitted);
    let retry = match definitely_not_submitted.dispatch() {
        Err(OperationDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("expected definitely-not-submitted retry authority, got {other:?}"),
    };
    let prior_attempt = retry.prior_attempt();
    let retry_invocation = retry.retry().unwrap();
    check(
        passed,
        retry_invocation.batch_invocation_id() != prior_attempt,
    );
    check(
        passed,
        definitely_not_submitted.reaper.retained_count() == 0,
    );
    definitely_not_submitted.set_submit_behavior(SubmitBehavior::Success);
    let retry_completion = definitely_not_submitted
        .dispatch_invocation(retry_invocation)
        .unwrap();
    assert!(matches!(
        retry_completion.poll(),
        Ok(CompletionObservation::Terminal(_))
    ));
    definitely_not_submitted.finish(passed);

    let pending = CompletionHarness::new();
    pending.set_fence_behavior(FenceBehavior::Pending);
    let handle = pending.dispatch().unwrap();
    let observer = handle.clone();
    drop(handle);
    check(passed, pending.reaper.retained_count() == 1);
    check(
        passed,
        matches!(observer.poll(), Ok(CompletionObservation::Pending)),
    );
    check(passed, pending.lane.in_flight_count() == 1);
    pending.set_fence_behavior(FenceBehavior::Succeeded);
    check(
        passed,
        matches!(observer.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    pending.finish(passed);

    let indeterminate = CompletionHarness::new();
    indeterminate.set_fence_behavior(FenceBehavior::Panic);
    let handle = indeterminate.dispatch().unwrap();
    let slot_id = handle.slot_id();
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.poll()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, indeterminate.reaper.retained_count() == 1);
    check(
        passed,
        indeterminate
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    check(passed, indeterminate.lane.is_reusable());
    indeterminate.set_fence_behavior(FenceBehavior::Indeterminate);
    let failure = match handle.poll().unwrap() {
        CompletionObservation::Indeterminate(failure) => failure,
        other => panic!("indeterminate fence produced {other:?}"),
    };
    check(
        passed,
        failure.len() == 1 && failure[0].failure().code() == "test_runtime",
    );
    check(
        passed,
        failure[0].failure().message() == "fence-indeterminate",
    );
    check(passed, indeterminate.reaper.retained_count() == 1);
    indeterminate.set_fence_behavior(FenceBehavior::FailedButQuiescent);
    let terminal = match handle.wait().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("failed-but-quiescent fence was not terminal: {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::FailedButQuiescent(failures) if failures.len() == 1 && failures[0].failure().code() == "test_runtime" && failures[0].failure().message() == "terminal-failure"),
    );
    indeterminate.finish(passed);

    let classification_panic = CompletionHarness::new();
    classification_panic.set_fence_behavior(FenceBehavior::Indeterminate);
    classification_panic.set_describe_error_panics(true);
    let handle = classification_panic.dispatch().unwrap();
    let slot_id = handle.slot_id();
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.poll()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, classification_panic.reaper.retained_count() == 1);
    check(passed, classification_panic.lane.is_reusable());
    check(
        passed,
        classification_panic
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    classification_panic.set_fence_behavior(FenceBehavior::FailedButQuiescent);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.wait()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, classification_panic.reaper.retained_count() == 1);
    classification_panic.set_describe_error_panics(false);
    let drain = match classification_panic
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => {
            panic!("classification panic recovery unexpectedly quarantined")
        }
    };
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceObservationPanicked
            && drain.had_submission_fence(),
    );
    check(passed, classification_panic.reaper.retained_count() == 0);
    check(passed, classification_panic.lane.is_fail_closed());
    classification_panic.finish(passed);

    let terminal_drift = CompletionHarness::new();
    let handle = terminal_drift.dispatch().unwrap();
    terminal_drift.drift_after_descriptor_reads(2);
    let terminal = match handle.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("terminal accounting drift produced {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::ContractFailedButQuiescent(failure) if failure.reason().contains("terminal accounting")),
    );
    check(passed, terminal_drift.lane.is_fail_closed());
    check(passed, terminal_drift.reaper.retained_count() == 0);
    check(passed, terminal_drift.lane.in_flight_count() == 0);
    terminal_drift
        .runtime
        .use_alternate_descriptor
        .store(false, Ordering::Release);
    terminal_drift.finish(passed);

    let terminal_stream_failure = CompletionHarness::new();
    let handle = terminal_stream_failure.dispatch().unwrap();
    terminal_stream_failure.set_stream_failed(true);
    let terminal = match handle.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("terminal stream failure produced {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::ContractFailedButQuiescent(failure) if failure.reason().contains("stream entered failed state")),
    );
    check(passed, terminal_stream_failure.lane.is_fail_closed());
    check(passed, terminal_stream_failure.reaper.retained_count() == 0);
    check(passed, terminal_stream_failure.lane.in_flight_count() == 0);
    terminal_stream_failure.set_stream_failed(false);
    terminal_stream_failure.finish(passed);

    let mut multiple = CompletionHarness::new();
    let second_resources = logical_resources(
        &multiple.plan_resources,
        "run.device-operation.completion.second",
        "request.device-operation.completion.second",
    );
    let second_session = second_resources.open_session().unwrap();
    let second_active = TrustedActiveSequenceBinding::from_session(&second_session).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();
    let second_step = begin_single_participant_step(&multiple.plan_resources, &second_batch);
    let first = multiple.dispatch().unwrap();
    let node_id = multiple.plan.payload().nodes()[0].id().clone();
    let provider = multiple
        .registry
        .bind(&multiple.resolved, &node_id)
        .unwrap();
    let second_frame_id = second_step.participant_frames().next().unwrap().frame_id();
    let second_invocation_id = NodeInvocationId::try_from(98).unwrap();
    let second_identity = operation_identity(
        &multiple.plan,
        &second_active,
        second_frame_id,
        second_invocation_id,
    );
    let second = encode_and_submit_single(
        &provider,
        &multiple.resolved,
        &second_identity,
        &second_frame_id,
        &second_invocation_id,
        &node_id,
        &second_active,
        admit_single_participant_invocation(&multiple.plan_resources, &second_step, &node_id),
        &multiple.lane,
        &multiple.reaper,
    )
    .unwrap();
    let first_slot = first.slot_id();
    let second_slot = second.slot_id();
    check(passed, first_slot.get() < second_slot.get());
    check(passed, multiple.lane.in_flight_count() == 2);
    check(passed, multiple.reaper.retained_count() == 2);
    multiple.set_fence_behavior_for(1, FenceBehavior::Pending);
    multiple.set_fence_behavior_for(2, FenceBehavior::Succeeded);
    let first_sweep = multiple.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        first_sweep.entries().len() == 1 && first_sweep.entries()[0].slot_id() == first_slot,
    );
    check(
        passed,
        matches!(
            first_sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Pending)
        ),
    );
    check(passed, first_sweep.retained_after() == 2);
    let second_sweep = multiple.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        second_sweep.entries().len() == 1 && second_sweep.entries()[0].slot_id() == second_slot,
    );
    check(
        passed,
        matches!(second_sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(passed, multiple.lane.in_flight_count() == 1);
    check(passed, multiple.reaper.retained_count() == 1);
    check(
        passed,
        matches!(first.poll(), Ok(CompletionObservation::Pending)),
    );
    multiple.set_fence_behavior_for(1, FenceBehavior::FailedButQuiescent);
    check(
        passed,
        matches!(first.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::FailedButQuiescent(_))),
    );
    check(passed, multiple.lane.in_flight_count() == 0);
    check(passed, multiple.reaper.retained_count() == 0);

    drop(second);
    drop(first);
    multiple
        .step
        .take()
        .expect("multiple-slot harness owns its first step")
        .try_retire_normal()
        .expect("first multiple-slot step is quiescent");
    multiple.step = Some(begin_single_participant_step(
        &multiple.plan_resources,
        &multiple.batch,
    ));
    assert!(second_step.try_retire_normal().is_ok());
    let second_step = begin_single_participant_step(&multiple.plan_resources, &second_batch);
    multiple.set_fence_behavior_for(3, FenceBehavior::Indeterminate);
    multiple.set_fence_behavior_for(4, FenceBehavior::Succeeded);
    let drain_target = multiple.dispatch().unwrap();
    let second_frame_id = second_step.participant_frames().next().unwrap().frame_id();
    let second_invocation_id = NodeInvocationId::try_from(100).unwrap();
    let second_identity = operation_identity(
        &multiple.plan,
        &second_active,
        second_frame_id,
        second_invocation_id,
    );
    let drain_sibling = encode_and_submit_single(
        &provider,
        &multiple.resolved,
        &second_identity,
        &second_frame_id,
        &second_invocation_id,
        &node_id,
        &second_active,
        admit_single_participant_invocation(&multiple.plan_resources, &second_step, &node_id),
        &multiple.lane,
        &multiple.reaper,
    )
    .unwrap();
    let drain_target_slot = drain_target.slot_id();
    check(
        passed,
        multiple.lane.in_flight_count() == 2 && multiple.reaper.retained_count() == 2,
    );
    check(
        passed,
        matches!(drain_target.poll(), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    check(
        passed,
        matches!(drain_target.wait(), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    let drain = match multiple
        .reaper
        .recover_slot_by_draining_lane(drain_target_slot)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => {
            panic!("selective lane drain unexpectedly quarantined")
        }
    };
    check(
        passed,
        drain.slot_id() == drain_target_slot
            && drain.cause() == CompletionRecoveryCause::FenceIndeterminate
            && drain.had_submission_fence(),
    );
    check(
        passed,
        multiple.lane.is_fail_closed() && multiple.lane.in_flight_count() == 1,
    );
    check(passed, multiple.reaper.retained_count() == 1);
    check(
        passed,
        multiple.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(
        passed,
        matches!(drain_sibling.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(
        passed,
        multiple.lane.in_flight_count() == 0 && multiple.reaper.retained_count() == 0,
    );
    drop(drain_sibling);
    drop(drain_target);
    check(passed, second_step.try_retire_normal().is_ok());
    check(passed, second_session.try_complete().is_ok());
    drop(second_batch);
    drop(second_session);
    drop(second_resources);
    multiple.finish(passed);

    let detached = CompletionHarness::new();
    detached.set_fence_behavior(FenceBehavior::Pending);
    let handle = detached.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    check(passed, detached.reaper.retained_count() == 1);
    check(passed, detached.reaper.poll_bounded(0).is_err());
    check(
        passed,
        detached
            .reaper
            .poll_bounded(MAX_COMPLETION_SWEEP_SLOTS + 1)
            .is_err(),
    );
    detached.set_fence_behavior(FenceBehavior::Succeeded);
    let sweep = detached.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        sweep.entries().len() == 1 && sweep.entries()[0].slot_id() == slot_id,
    );
    check(
        passed,
        matches!(sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(passed, sweep.retained_after() == 0);
    check(passed, sweep.quarantined_after() == 0);
    check(passed, detached.lane.in_flight_count() == 0);
    detached.finish(passed);

    let recovered = CompletionHarness::new();
    recovered.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = recovered.dispatch().unwrap();
    let expected_identity = handle.batch_identity().clone();
    drop(handle);
    let sweep = recovered.reaper.poll_bounded(1).unwrap();
    check(passed, sweep.entries().len() == 1);
    let slot_id = sweep.entries()[0].slot_id();
    check(
        passed,
        matches!(
            sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Indeterminate(_))
        ),
    );
    check(
        passed,
        recovered
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    check(passed, recovered.lane.is_reusable());
    check(passed, recovered.reaper.retained_count() == 1);
    check(
        passed,
        matches!(recovered.reaper.wait_slot_for_recovery(slot_id), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    let drain = match recovered
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("successful drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(passed, drain.batch_identity() == &expected_identity);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(passed, drain.had_submission_fence());
    check(
        passed,
        recovered.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, recovered.reaper.retained_count() == 0);
    check(passed, recovered.lane.in_flight_count() == 0);
    check(passed, recovered.lane.is_fail_closed());
    let failed_lane_resources = logical_resources(
        &recovered.plan_resources,
        "run.device-operation.completion.failed-lane",
        "request.device-operation.completion.failed-lane",
    );
    let failed_lane_session = failed_lane_resources.open_session().unwrap();
    let failed_lane_active =
        TrustedActiveSequenceBinding::from_session(&failed_lane_session).unwrap();
    let failed_lane_batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&failed_lane_session)]).unwrap();
    let failed_lane_step =
        begin_single_participant_step(&recovered.plan_resources, &failed_lane_batch);
    let failed_lane_frame = failed_lane_step
        .participant_frames()
        .next()
        .unwrap()
        .frame_id();
    let failed_lane_invocation_id = NodeInvocationId::try_from(101).unwrap();
    let failed_lane_identity = operation_identity(
        &recovered.plan,
        &failed_lane_active,
        failed_lane_frame,
        failed_lane_invocation_id,
    );
    let failed_lane_node = &recovered.plan.payload().nodes()[0];
    let failed_lane_provider = recovered
        .registry
        .bind(&recovered.resolved, failed_lane_node.id())
        .unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &failed_lane_provider,
                &recovered.resolved,
                &failed_lane_identity,
                &failed_lane_frame,
                &failed_lane_invocation_id,
                failed_lane_node.id(),
                &failed_lane_active,
                admit_single_participant_invocation(
                    &recovered.plan_resources,
                    &failed_lane_step,
                    failed_lane_node.id(),
                ),
                &recovered.lane,
                &recovered.reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(
        passed,
        recovered.runtime_trace.lock().unwrap().submit_calls == 1,
    );
    assert!(failed_lane_step.try_retire_normal().is_ok());
    assert!(failed_lane_session.try_complete().is_ok());
    drop(failed_lane_batch);
    drop(failed_lane_session);
    drop(failed_lane_resources);
    recovered.finish(passed);

    let wait_panic = CompletionHarness::new();
    wait_panic.set_fence_behavior(FenceBehavior::Panic);
    let handle = wait_panic.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| wait_panic.reaper.wait_slot_for_recovery(slot_id)),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    let drain = match wait_panic
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("wait panic drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceObservationPanicked,
    );
    check(passed, drain.had_submission_fence());
    check(
        passed,
        wait_panic.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, wait_panic.reaper.retained_count() == 0);
    check(passed, wait_panic.lane.is_fail_closed());
    wait_panic.finish(passed);

    let quarantined = CompletionHarness::new();
    quarantined.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = quarantined.dispatch().unwrap();
    let expected_identity = handle.batch_identity().clone();
    drop(handle);
    let sweep = quarantined.reaper.poll_bounded(1).unwrap();
    check(passed, sweep.entries().len() == 1);
    let slot_id = sweep.entries()[0].slot_id();
    check(
        passed,
        matches!(
            sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Indeterminate(_))
        ),
    );
    check(
        passed,
        matches!(
            quarantined.reaper.wait_slot_for_recovery(slot_id),
            Ok(CompletionObservation::Indeterminate(_))
        ),
    );
    quarantined.set_synchronize_fails(true);
    let quarantine = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Quarantined(receipt) => receipt,
        CompletionRecoveryOutcome::Drained(_) => panic!("failed drain released ownership"),
    };
    let stale_quarantine = quarantine.clone();
    check(
        passed,
        quarantine.is_current() && stale_quarantine.is_current(),
    );
    check(
        passed,
        serde_json::to_value(&quarantine)
            .unwrap()
            .get("freshness")
            .is_none(),
    );
    check(passed, quarantine.slot_id() == slot_id);
    check(passed, quarantine.batch_identity() == &expected_identity);
    check(
        passed,
        quarantine.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(passed, quarantine.had_submission_fence());
    check(
        passed,
        quarantine.device_id() == &quarantined.lane.descriptor().id,
    );
    check(
        passed,
        quarantine.runtime_implementation_fingerprint()
            == quarantined
                .lane
                .descriptor()
                .runtime_implementation_fingerprint,
    );
    check(passed, quarantined.reaper.retained_count() == 1);
    check(passed, quarantined.reaper.quarantined_count() == 1);
    check(passed, quarantined.lane.is_fail_closed());
    let sweep = quarantined.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        matches!(sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Quarantined(receipt)) if receipt == &quarantine),
    );
    check(
        passed,
        sweep.retained_after() == 1 && sweep.quarantined_after() == 1,
    );
    quarantined.set_synchronize_fails(false);
    let drain = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("retry drain remained quarantined"),
    };
    check(
        passed,
        drain.slot_id() == slot_id && drain.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(
        passed,
        !quarantine.is_current() && !stale_quarantine.is_current(),
    );
    check(
        passed,
        quarantined.runtime_trace.lock().unwrap().synchronize_calls == 2,
    );
    check(passed, quarantined.reaper.retained_count() == 0);
    check(passed, quarantined.reaper.quarantined_count() == 0);
    check(passed, quarantined.lane.in_flight_count() == 0);
    quarantined.finish(passed);

    let submit_panic = CompletionHarness::new();
    submit_panic.set_submit_behavior(SubmitBehavior::Panic);
    let recovery = match suppress_expected_panic_hook(|| submit_panic.dispatch()) {
        Err(OperationDispatchError::SubmissionIndeterminate { recovery }) => recovery,
        _ => panic!("submit panic did not retain recovery ownership"),
    };
    let slot_id = recovery.slot_id();
    check(passed, submit_panic.reaper.retained_count() == 1);
    let drain = match recovery.recover_by_draining_lane().unwrap() {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("submit panic drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::SubmissionIndeterminate,
    );
    check(passed, !drain.had_submission_fence());
    check(
        passed,
        submit_panic.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, submit_panic.reaper.retained_count() == 0);
    check(passed, submit_panic.lane.is_fail_closed());
    submit_panic.finish(passed);

    let drop_fallback = CompletionHarness::new();
    drop_fallback.set_submit_behavior(SubmitBehavior::Panic);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| drop_fallback.dispatch()),
            Err(OperationDispatchError::SubmissionIndeterminate { .. })
        ),
    );
    check(passed, drop_fallback.reaper.retained_count() == 1);
    let CompletionHarness {
        reaper,
        lane,
        step,
        session,
        batch,
        resources,
        runtime,
        plan_resources,
        ..
    } = drop_fallback;
    drop(reaper);
    let cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    check(
        passed,
        retire_step_after_deferred_cleanup(step.expect("harness owns a step")).is_ok()
            && lane.in_flight_count() == 0,
    );
    session.try_complete().unwrap();
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    check(
        passed,
        matches!(
            PlanRuntimeResources::close(plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ),
    );
    assert_eq!(*passed - start, EXPECTED_COMPLETION_CASES);
}

fn wrong_runtime_admission_contract(passed: &mut usize) {
    let catalog = catalog();
    let behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let provider_trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = operation_registry(&catalog, behavior, provider_trace);
    let plan = plan_for_registry(&registry);
    let (runtime, runtime_trace) = runtime(&catalog);
    runtime
        .use_alternate_descriptor
        .store(true, Ordering::Release);
    check(
        passed,
        plan.provision_static(runtime, id("request.device-operation.wrong-runtime"))
            .is_err(),
    );
    check(passed, runtime_trace.lock().unwrap().allocation_calls == 0);
}

fn admit_batch_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![one_token_span(); batch.len() as usize])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            _ => panic!("batch step admission did not converge"),
        }
    }
    unreachable!("bounded batch step admission returns or panics")
}

fn admit_batch_invocation(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![
            one_token_span();
            step.participant_count() as usize
        ])
        .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => return invocation,
            InvocationResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            _ => panic!("batch invocation admission did not converge"),
        }
    }
    unreachable!("bounded batch invocation admission returns or panics")
}

fn thirty_two_participant_dispatch_is_one_physical_submission() {
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture();
    let resources = (0..32)
        .map(|index| {
            logical_resources(
                &plan_resources,
                &format!("run.device-operation.batch32.{index}"),
                &format!("request.device-operation.batch32.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let step = admit_batch_step(&plan_resources, &batch);
    let node = &plan.payload().nodes()[0];
    let invocation = admit_batch_invocation(&plan_resources, &step, node.id());
    let identities = step
        .participant_frames()
        .zip(&active_bindings)
        .enumerate()
        .map(|(index, (frame, active))| {
            operation_identity(
                &plan,
                active,
                frame.frame_id(),
                NodeInvocationId::try_from(index as u64 + 1).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let batch_identity = OperationDispatch::bind_batch_identity(
        &resolved,
        identities,
        &active_bindings,
        &invocation,
        &lane,
    )
    .unwrap();
    assert_eq!(batch_identity.participants().len(), 32);
    assert_eq!(
        batch_identity.work_shape_fingerprint(),
        invocation.work_shape().fingerprint()
    );
    let handle = OperationDispatch::encode_and_submit(
        &provider,
        &resolved,
        &batch_identity,
        &active_bindings,
        invocation,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(runtime_trace.lock().unwrap().submit_calls, 1);
    let trace = provider_trace.lock().unwrap();
    assert_eq!(trace.encode_calls, 1);
    assert_eq!(trace.last_participant_count, 32);
    assert_eq!(trace.last_work_sequences, 32);
    drop(trace);
    assert_eq!(handle.receipt().participants().len(), 32);
    assert!(handle
        .receipt()
        .participants()
        .iter()
        .all(|participant| participant.batch_submission_fingerprint()
            == handle.receipt().fingerprint()));
    let completion = match handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("32-participant batch did not complete: {other:?}"),
    };
    assert_eq!(completion.participants().len(), 32);
    assert!(completion
        .participants()
        .iter()
        .all(|participant| participant.batch_completion_fingerprint() == completion.fingerprint()));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(resources);
    drop(reaper);
    drop(lane);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn device_and_operation_contract_is_exhaustive() {
    let mut passed = 0;
    descriptor_and_registry_contract(&mut passed);
    wrong_runtime_admission_contract(&mut passed);
    operation_dispatch_contract(fixture(), &mut passed);
    cancel_dispatch_linearization_contract(fixture(), &mut passed);
    submit_descriptor_drift_contract(fixture(), &mut passed);
    legacy_source_seals_sequence_and_cannot_authorize_other_session(fixture(), &mut passed);
    completion_reaper_owns_invocations_until_quiescent_terminal(&mut passed);
    thirty_two_participant_dispatch_is_one_physical_submission();
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT DEVICE OPERATION PASS: {passed}/{EXPECTED_CASES}");
}
