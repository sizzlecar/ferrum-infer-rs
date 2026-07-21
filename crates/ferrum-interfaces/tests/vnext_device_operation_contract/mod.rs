pub(crate) use ferrum_interfaces::vnext::*;
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::{json, Value};
pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::error::Error;
pub(crate) use std::fmt;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
pub(crate) use std::sync::{mpsc, Arc, Barrier, Mutex};
pub(crate) use std::time::{Duration, Instant};

pub(crate) const EXPECTED_LEGACY_AUTHORITY_CASES: usize = 13;
pub(crate) const EXPECTED_CANCEL_DISPATCH_CASES: usize = 16;
pub(crate) const EXPECTED_COMPLETION_CASES: usize = 200;
pub(crate) const EXPECTED_CASES: usize = 299;
pub(crate) const COMPLETION_DROP_TEST_WORKERS: usize = 1;
pub(crate) const MAX_COMPLETION_DROP_TEST_WORKERS: usize = 2;
pub(crate) const _: () = assert!(
    COMPLETION_DROP_TEST_WORKERS == 1
        && COMPLETION_DROP_TEST_WORKERS <= MAX_COMPLETION_DROP_TEST_WORKERS
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

pub(crate) fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}

pub(crate) fn suppress_expected_panic_hook<T>(action: impl FnOnce() -> T) -> T {
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
pub(crate) struct TestConfig {
    pub(crate) width: u64,
    #[serde(default)]
    pub(crate) zero_state: bool,
}

#[derive(Default)]
pub(crate) struct TestFamily;

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
        let mut main_inputs = vec![id("value.input"), id("value.weight")];
        let mut tail_inputs = vec![id("value.intermediate"), id("value.weight")];
        let states = if config.zero_state {
            main_inputs.push(id("value.state"));
            tail_inputs.push(id("value.state"));
            vec![StateSpec {
                id: id("state.device-operation"),
                value_id: id("value.state"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
                initialization: StateInitialization::Zero,
            }]
        } else {
            Vec::new()
        };
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.main"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: main_inputs,
                        outputs: vec![id("value.intermediate")],
                        attributes: BTreeMap::new(),
                    },
                    ProgramNode {
                        id: id("node.tail"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: tail_inputs,
                        outputs: vec![id("value.output")],
                        attributes: BTreeMap::new(),
                    },
                ],
            }],
            states,
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

pub(crate) fn tensor_contract(access: TensorAccess) -> TensorContract {
    typed_tensor_contract(ElementType::F32, access)
}

pub(crate) fn typed_tensor_contract(
    element_type: ElementType,
    access: TensorAccess,
) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(4)],
        BTreeSet::from([element_type]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

pub(crate) fn operation() -> OperationDescriptor {
    operation_with_zero_state(false)
}

pub(crate) fn operation_with_zero_state(zero_state: bool) -> OperationDescriptor {
    let mut inputs = vec![
        tensor_contract(TensorAccess::Read),
        tensor_contract(TensorAccess::Read),
    ];
    if zero_state {
        inputs.push(typed_tensor_contract(ElementType::U8, TensorAccess::Read));
    }
    OperationDescriptor {
        id: id("operation.main"),
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![tensor_contract(TensorAccess::Write)],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
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

pub(crate) fn catalog() -> CapabilityCatalog {
    catalog_with_zero_state(false)
}

pub(crate) fn catalog_with_zero_state(zero_state: bool) -> CapabilityCatalog {
    let operation = operation_with_zero_state(zero_state);
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

pub(crate) struct TestOperationContract {
    pub(crate) descriptor: OperationDescriptor,
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
pub(crate) enum ProviderBehavior {
    Success,
    SplitPhases,
    WrongIdentity,
    WrongPhase,
}

#[derive(Default)]
pub(crate) struct ProviderTrace {
    pub(crate) encode_calls: u64,
    pub(crate) last_participant_count: usize,
    pub(crate) last_work_sequences: u32,
    pub(crate) component_resources: BTreeSet<ResourceId>,
    pub(crate) view_resources: BTreeSet<ResourceId>,
}

pub(crate) struct TestProvider {
    pub(crate) descriptor: OperationProviderDescriptor,
    pub(crate) behavior: Arc<Mutex<ProviderBehavior>>,
    pub(crate) trace: Arc<Mutex<ProviderTrace>>,
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
    ) -> Result<EncodedDeviceOperation<TestCommand>, OperationFailure> {
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
            ProviderBehavior::Success => Ok(EncodedDeviceOperation::compute(TestCommand::Provider)),
            ProviderBehavior::SplitPhases => {
                Ok(EncodedDeviceOperation::compute(TestCommand::Provider)
                    .with_dynamic_binding(TestCommand::DynamicBinding)
                    .with_result_binding(TestCommand::ResultBinding))
            }
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

pub(crate) fn policy() -> ResolvedRuntimePolicy {
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
            maximum_scheduled_tokens: 4096,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        None,
    )
    .unwrap()
}

pub(crate) fn resolved_tensor() -> ResolvedTensorSpec {
    resolved_tensor_for(ElementType::F32)
}

pub(crate) fn resolved_tensor_for(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

fn resolved_weight() -> ResolvedWeightBinding {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    ResolvedWeightBinding::from_schema(family.weight_schema(), &id("weight.matrix")).unwrap()
}

pub(crate) fn single_binding(
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
        None,
        ResolvedValueStorage::single(id(resource), 0, 16, ElementType::F32).unwrap(),
    )
    .unwrap()
}

pub(crate) fn node_values_for(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
) -> Vec<ResolvedValueBinding> {
    vec![
        single_binding(
            input_value,
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            input_resource,
        ),
        ResolvedValueBinding::new(
            id("value.weight"),
            ResolvedValueRole::Input,
            1,
            resolved_tensor(),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Weights,
            Some(resolved_weight()),
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
            output_value,
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            output_resource,
        ),
    ]
}

pub(crate) fn node_values_with_zero_state_for(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
) -> Vec<ResolvedValueBinding> {
    let mut values = node_values_for(input_value, input_resource, output_value, output_resource);
    values.insert(
        2,
        ResolvedValueBinding::new(
            id("value.state"),
            ResolvedValueRole::Input,
            2,
            resolved_tensor_for(ElementType::U8),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::State,
            None,
            ResolvedValueStorage::single(id("resource.state"), 0, 4, ElementType::U8).unwrap(),
        )
        .unwrap(),
    );
    values
}

pub(crate) fn node_values() -> Vec<ResolvedValueBinding> {
    node_values_for(
        "value.input",
        "resource.input",
        "value.intermediate",
        "resource.intermediate",
    )
}

pub(crate) fn tail_node_values() -> Vec<ResolvedValueBinding> {
    node_values_for(
        "value.intermediate",
        "resource.intermediate",
        "value.output",
        "resource.output",
    )
}

#[derive(Debug)]
pub(crate) struct TestBuffer {
    pub(crate) descriptor: BufferDescriptor,
}

#[derive(Debug, Default)]
pub(crate) struct TestStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TestCommand {
    Provider,
    DynamicBinding,
    ResultBinding,
    Copy,
    Upload,
    Zero,
}

#[derive(Debug, Clone)]
pub(crate) struct TestFence(u64, DeviceTimingMode, Option<DeviceSubmissionAttribution>);

impl TestFence {
    fn terminal_receipt(
        &self,
        terminal: DeviceTerminal<TestRuntimeError>,
    ) -> DeviceTerminalReceipt<TestRuntimeError> {
        match self.1 {
            DeviceTimingMode::Off => DeviceTerminalReceipt::unprofiled(terminal),
            DeviceTimingMode::Completion | DeviceTimingMode::Kernel => {
                DeviceTerminalReceipt::profiled(
                    terminal,
                    DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(
                        1_000_000,
                    )),
                )
            }
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum SubmitBehavior {
    #[default]
    Success,
    DefinitelyNotSubmitted,
    Panic,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum FenceBehavior {
    Pending,
    #[default]
    Succeeded,
    FailedButQuiescent,
    Indeterminate,
    Panic,
}

#[derive(Default)]
pub(crate) struct RuntimeTrace {
    pub(crate) allocation_calls: u64,
    pub(crate) submit_calls: u64,
    pub(crate) submitted_command_counts: Vec<usize>,
    pub(crate) submitted_command_phases: Vec<Vec<DeviceCommandPhase>>,
    pub(crate) submitted_commands: Vec<Vec<TestCommand>>,
    pub(crate) readback_calls: u64,
    pub(crate) readback_lengths: Vec<u64>,
    pub(crate) synchronize_calls: u64,
    pub(crate) wait_fence_calls: u64,
    pub(crate) tamper_buffer_descriptor: bool,
    pub(crate) drift_on_submit: bool,
    pub(crate) next_fence: u64,
    pub(crate) submit_behavior: SubmitBehavior,
    pub(crate) fence_behavior: FenceBehavior,
    pub(crate) fence_behaviors: BTreeMap<u64, FenceBehavior>,
    pub(crate) wait_fence_block: Option<(Arc<Barrier>, Arc<Barrier>)>,
    pub(crate) synchronize_fails: bool,
    pub(crate) stream_failed: bool,
    pub(crate) describe_error_panics: bool,
    pub(crate) static_weight_import_enabled: bool,
    pub(crate) static_weight_import_begin_calls: u64,
    pub(crate) static_weight_import_seal_calls: u64,
    pub(crate) imported_component_count: usize,
    pub(crate) imported_bytes: u64,
}

pub(crate) struct TestRuntime {
    pub(crate) descriptor: DeviceDescriptor,
    pub(crate) alternate_descriptor: DeviceDescriptor,
    pub(crate) use_alternate_descriptor: AtomicBool,
    pub(crate) descriptor_reads_until_drift: AtomicU64,
    pub(crate) trace: Arc<Mutex<RuntimeTrace>>,
}

struct TestStaticWeightImport {
    trace: Arc<Mutex<RuntimeTrace>>,
    components: Vec<(ResourceId, u64, u64)>,
}

impl StaticWeightImportSession<TestBuffer, TestRuntimeError> for TestStaticWeightImport {
    fn import_component(
        &mut self,
        payload: &WeightComponentPayload<'_>,
        destination: &TestBuffer,
        destination_offset_bytes: u64,
    ) -> Result<(), TestRuntimeError> {
        let length_bytes = u64::try_from(payload.bytes().len())
            .map_err(|_| TestRuntimeError("import payload exceeds u64"))?;
        let end = destination_offset_bytes
            .checked_add(length_bytes)
            .ok_or(TestRuntimeError("import range overflows"))?;
        if destination.descriptor.usage != BufferUsage::Weights
            || destination.descriptor.element_type != payload.element_type()
            || end > destination.descriptor.size_bytes
        {
            return Err(TestRuntimeError("import range differs from destination"));
        }
        self.components.push((
            destination.descriptor.resource_id.clone(),
            destination_offset_bytes,
            length_bytes,
        ));
        Ok(())
    }

    fn seal(self: Box<Self>) -> Result<(), TestRuntimeError> {
        let imported_bytes = self
            .components
            .iter()
            .try_fold(0_u64, |total, (_, _, length)| total.checked_add(*length))
            .ok_or(TestRuntimeError("import byte count overflows"))?;
        let mut trace = self.trace.lock().unwrap();
        trace.static_weight_import_seal_calls += 1;
        trace.imported_component_count += self.components.len();
        trace.imported_bytes += imported_bytes;
        Ok(())
    }
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

    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        let mut trace = self.trace.lock().unwrap();
        if !trace.static_weight_import_enabled {
            return None;
        }
        trace.static_weight_import_begin_calls += 1;
        Some(Ok(Box::new(TestStaticWeightImport {
            trace: Arc::clone(&self.trace),
            components: Vec::new(),
        })))
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
        Ok(TestCommand::Copy)
    }

    fn encode_upload(
        &self,
        _source: &[u8],
        _source_layout: HostTransferLayout,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand::Upload)
    }

    fn encode_zero(
        &self,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(TestCommand::Zero)
    }

    fn submit(
        &self,
        _stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        assert!(!commands.is_empty(), "core must not submit an empty batch");
        let timing_mode = commands.timing_mode();
        let entries = commands.into_entries();
        let command_phases = entries.iter().map(DeviceCommandEntry::phase).collect();
        let attribution = timing_mode.kernel_attribution_enabled().then(|| {
            entries
                .iter()
                .filter_map(|entry| {
                    let (native_op_id, compute_dispatch_count, transfer_command_count) =
                        match entry.command() {
                            TestCommand::Provider => ("test_provider", 1, 0),
                            TestCommand::DynamicBinding => ("test_dynamic_binding", 0, 1),
                            TestCommand::ResultBinding => ("test_result_binding", 0, 1),
                            TestCommand::Copy => ("test_copy", 0, 1),
                            TestCommand::Upload => ("test_upload", 0, 1),
                            TestCommand::Zero => ("test_zero", 0, 1),
                        };
                    DeviceNativeWorkAttribution::new(
                        entry.node_index(),
                        entry.phase(),
                        native_op_id,
                        DeviceExecutionPath::Eager,
                        DeviceBatchingForm::Scalar,
                        u32::from(entry.node_index().is_some()),
                        u64::from(entry.node_index().is_some()),
                        compute_dispatch_count,
                        transfer_command_count,
                    )
                })
                .collect()
        });
        let attribution = attribution.and_then(DeviceSubmissionAttribution::new);
        let commands = entries
            .into_iter()
            .map(DeviceCommandEntry::into_parts)
            .map(|(_, _, command)| command)
            .collect::<Vec<_>>();
        let command_count = commands.len();
        let (drift, behavior, fence) = {
            let mut trace = self.trace.lock().unwrap();
            trace.submit_calls += 1;
            trace.submitted_command_counts.push(command_count);
            trace.submitted_command_phases.push(command_phases);
            trace.submitted_commands.push(commands);
            trace.next_fence += 1;
            (
                trace.drift_on_submit,
                trace.submit_behavior,
                TestFence(trace.next_fence, timing_mode, attribution),
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

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        fence.2.clone()
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
            FenceBehavior::Succeeded => {
                FenceQuery::Terminal(fence.terminal_receipt(DeviceTerminal::Succeeded))
            }
            FenceBehavior::FailedButQuiescent => FenceQuery::Terminal(fence.terminal_receipt(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError("terminal-failure")),
            )),
            FenceBehavior::Indeterminate => {
                FenceQuery::Indeterminate(TestRuntimeError("fence-indeterminate"))
            }
            FenceBehavior::Panic => panic!("injected query panic"),
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
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
            FenceBehavior::Succeeded => Ok(fence.terminal_receipt(DeviceTerminal::Succeeded)),
            FenceBehavior::FailedButQuiescent => Ok(fence.terminal_receipt(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError("terminal-failure")),
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
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        let mut trace = self.trace.lock().unwrap();
        trace.readback_calls += 1;
        trace.readback_lengths.push(region.length_bytes());
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
pub(crate) struct DriverTrace {
    pub(crate) calls: u64,
}

pub(crate) struct TestDriver {
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) trace: Arc<Mutex<DriverTrace>>,
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

pub(crate) fn resource_failure(code: &str) -> ResourceDriverFailure {
    ResourceDriverFailure::new(
        FailureEnvelope::new(FailureDomain::Resource, code, "resource failure", false).unwrap(),
    )
    .unwrap()
}

pub(crate) fn runtime(catalog: &CapabilityCatalog) -> (Arc<TestRuntime>, Arc<Mutex<RuntimeTrace>>) {
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

pub(crate) struct TestModelRegistry {
    pub(crate) registration: TypedFamilyRegistration<TestFamily>,
}

impl TestModelRegistry {
    pub(crate) fn new() -> Self {
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

pub(crate) const RESOLUTION_FIELDS: [ResolutionField; 20] = [
    ResolutionField::OriginalSources,
    ResolutionField::ResolvedSources,
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

pub(crate) fn resolution_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSources => ResolutionDecisionSource::UserInput,
        ResolutionField::ResolvedSources
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

pub(crate) fn resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
    match field {
        ResolutionField::OriginalSources => serde_json::to_value(&inputs.original_sources).unwrap(),
        ResolutionField::ResolvedSources => serde_json::to_value(&inputs.resolved_sources).unwrap(),
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

pub(crate) fn operation_registry(
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

pub(crate) fn node_resolution_for(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    node_id: &str,
    values: Vec<ResolvedValueBinding>,
) -> PlanNodeResolution {
    PlanNodeResolution::resolve(
        family,
        catalog,
        runtime_policy,
        &registry.planning(),
        id(node_id),
        values,
        BTreeSet::new(),
        None,
    )
    .unwrap()
}

pub(crate) fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> PlanNodeResolution {
    node_resolution_with_zero_state(family, catalog, runtime_policy, registry, false)
}

pub(crate) fn node_resolution_with_zero_state(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> PlanNodeResolution {
    node_resolution_for(
        family,
        catalog,
        runtime_policy,
        registry,
        "node.main",
        if zero_state {
            node_values_with_zero_state_for(
                "value.input",
                "resource.input",
                "value.intermediate",
                "resource.intermediate",
            )
        } else {
            node_values()
        },
    )
}

pub(crate) fn tail_node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> PlanNodeResolution {
    tail_node_resolution_with_zero_state(family, catalog, runtime_policy, registry, false)
}

pub(crate) fn tail_node_resolution_with_zero_state(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> PlanNodeResolution {
    node_resolution_for(
        family,
        catalog,
        runtime_policy,
        registry,
        "node.tail",
        if zero_state {
            node_values_with_zero_state_for(
                "value.intermediate",
                "resource.intermediate",
                "value.output",
                "resource.output",
            )
        } else {
            tail_node_values()
        },
    )
}

pub(crate) fn resolved_model_plan(
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> (ResolvedModelPlan, ExecutionPlan) {
    resolved_model_plan_with_zero_state(registry, false)
}

pub(crate) fn resolved_model_plan_with_zero_state(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> (ResolvedModelPlan, ExecutionPlan) {
    let model_registry = TestModelRegistry::new();
    let raw_config = if zero_state {
        json!({"width": 4, "zero_state": true})
    } else {
        json!({"width": 4})
    };
    let family = model_registry.registration.prepare(&raw_config).unwrap();
    let catalog = catalog_with_zero_state(zero_state);
    let runtime_policy = policy();
    let resolutions = vec![
        node_resolution_with_zero_state(&family, &catalog, &runtime_policy, registry, zero_state),
        tail_node_resolution_with_zero_state(
            &family,
            &catalog,
            &runtime_policy,
            registry,
            zero_state,
        ),
    ];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &runtime_policy, resolutions.clone()).unwrap(),
    )
    .unwrap();
    let config_fingerprint = family.config_fingerprint().to_owned();
    let original_source = OriginalModelSource {
        kind: ModelSourceKind::Repository,
        location: "repo/device-operation-model".to_owned(),
        requested_revision: Some("main".to_owned()),
    };
    let resolved_source = ResolvedModelSource {
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
    };
    let inputs = ResolvedModelPlanInputs {
        original_sources: OriginalModelSources {
            semantic: original_source.clone(),
            tokenizer: original_source.clone(),
            weights: original_source,
        },
        resolved_sources: ResolvedModelSources {
            semantic: resolved_source.clone(),
            tokenizer: resolved_source.clone(),
            weights: resolved_source,
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
            collision_policy: StopTokenCollisionPolicy::require_distinct(),
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

pub(crate) fn plan_for_registry(registry: &OperationRuntimeRegistry<TestRuntime>) -> ExecutionPlan {
    plan_for_registry_with_zero_state(registry, false)
}

pub(crate) fn plan_for_registry_with_zero_state(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> ExecutionPlan {
    let raw_config = if zero_state {
        json!({"width": 4, "zero_state": true})
    } else {
        json!({"width": 4})
    };
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&raw_config)
        .unwrap();
    let catalog = catalog_with_zero_state(zero_state);
    let runtime_policy = policy();
    ExecutionPlan::build(
        PlanBuildRequest::new(
            &family,
            &catalog,
            &runtime_policy,
            vec![
                node_resolution_with_zero_state(
                    &family,
                    &catalog,
                    &runtime_policy,
                    registry,
                    zero_state,
                ),
                tail_node_resolution_with_zero_state(
                    &family,
                    &catalog,
                    &runtime_policy,
                    registry,
                    zero_state,
                ),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}

pub(crate) fn plan_runtime_resources(
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

pub(crate) fn logical_resources(
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
                deferred.maintain().unwrap();
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
                deferred.maintain().unwrap();
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

pub(crate) fn begin_single_participant_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let lane = plan_resources.create_execution_lane().unwrap();
    begin_single_participant_step_on_lane(batch, &lane)
}

pub(crate) fn begin_single_participant_step_on_lane(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
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
                deferred.maintain().unwrap();
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

pub(crate) fn admit_single_participant_invocation(
    _plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
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
                deferred.maintain().unwrap();
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
pub(crate) fn encode_and_submit_single(
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

pub(crate) struct Fixture {
    pub(crate) registry: OperationRuntimeRegistry<TestRuntime>,
    pub(crate) impostor_registry: OperationRuntimeRegistry<TestRuntime>,
    pub(crate) resolved: ResolvedModelPlan,
    pub(crate) plan: ExecutionPlan,
    pub(crate) impostor_plan_hash: PlanHash,
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) runtime_trace: Arc<Mutex<RuntimeTrace>>,
    pub(crate) provider_behavior: Arc<Mutex<ProviderBehavior>>,
    pub(crate) provider_trace: Arc<Mutex<ProviderTrace>>,
    pub(crate) plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
}

pub(crate) fn fixture() -> Fixture {
    fixture_with_zero_state(false)
}

pub(crate) fn fixture_with_zero_state(zero_state: bool) -> Fixture {
    let catalog = catalog_with_zero_state(zero_state);
    let provider_behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let provider_trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = operation_registry(
        &catalog,
        Arc::clone(&provider_behavior),
        Arc::clone(&provider_trace),
    );
    let derived_catalog = registry
        .capability_catalog(
            catalog.device().clone(),
            catalog.engine_providers().values().cloned().collect(),
        )
        .unwrap();
    assert_eq!(derived_catalog, catalog);
    let (resolved, plan) = resolved_model_plan_with_zero_state(&registry, zero_state);
    let impostor_registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::WrongPhase)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let impostor_plan_hash = plan_for_registry_with_zero_state(&impostor_registry, zero_state)
        .plan_hash()
        .clone();
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

pub(crate) fn close_plan_runtime(
    plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
    passed: &mut usize,
) {
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

pub(crate) fn device_identity_parts(
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

pub(crate) fn operation_identity(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    frame_id: ExecutionFrameId,
    invocation_id: NodeInvocationId,
) -> ExecutionIdentityEnvelope {
    operation_identity_for_node(plan, 0, active, frame_id, invocation_id)
}

pub(crate) fn operation_identity_for_node(
    plan: &ExecutionPlan,
    node_index: usize,
    active: &TrustedActiveSequenceBinding,
    frame_id: ExecutionFrameId,
    invocation_id: NodeInvocationId,
) -> ExecutionIdentityEnvelope {
    let node = &plan.payload().nodes()[node_index];
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
        span_id: id(format!("span.device-operation.node.{node_index}")),
        parent_span_id: None,
        async_links: Vec::new(),
    })
    .unwrap()
}

pub(crate) fn revalidate_plan_for_registry(
    bytes: &[u8],
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let resolutions = vec![
        node_resolution(&family, &catalog, &runtime_policy, registry),
        tail_node_resolution(&family, &catalog, &runtime_policy, registry),
    ];
    ExecutionPlan::from_json_validated(bytes, &family, &catalog, &runtime_policy, resolutions)
        .unwrap()
}

pub(crate) fn serialization_message(error: VNextError) -> String {
    match error {
        VNextError::Serialization { message, .. } => message,
        other => panic!("expected serialization error, got {other}"),
    }
}
