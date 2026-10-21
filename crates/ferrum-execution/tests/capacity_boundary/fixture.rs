//! One-operation public-runtime fixture, adapted from the runtime resource
//! contract tests. It exercises allocation and lifecycle, not tensor numerics.

use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

static NEXT_DEVICE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(crate) fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

pub(crate) fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

pub(crate) fn fixture_f32_profiles(
    family: &ModelFamilyId,
    outputs: &[&str],
    operations: &[&str],
    states: Vec<StateSpec>,
) -> Result<FamilyNumericalProfiles, VNextError> {
    let profile_id = NumericalProfileId::new("fixture.f32").unwrap();
    FamilyNumericalProfiles::new(
        family,
        ContractVersion::new(1, 0),
        vec![NumericalExecutionProfile {
            kv_storage: Vec::new(),
            id: profile_id.clone(),
            family_id: family.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(*outputs.last().unwrap()),
            boundaries: outputs
                .iter()
                .map(|output| (id(*output), ElementType::F32))
                .collect(),
            states,
            operations: operations
                .iter()
                .map(|operation| NumericalOperationContract {
                    operation_id: id(*operation),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
                .collect(),
        }],
        vec![profile_id],
    )
}

pub(crate) fn fixture_byte_state(width: u64) -> StateSpec {
    StateSpec {
        id: id("state.cache"),
        value_id: id("value.state"),
        tensor: ProgramTensorSpec {
            dimensions: vec![width],
            element_type: ElementType::U8,
            layout: ResolvedTensorLayout::Contiguous,
        },
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::FixedPerScope,
        initialization: StateInitialization::Zero,
        checkpoint: StateCheckpointCapability::Unsupported,
    }
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

    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        fixture_f32_profiles(
            self.family_id(),
            &["value.output"],
            &["operation.main"],
            vec![fixture_byte_state(config.width)],
        )
    }

    fn semantic_program(
        &self,
        config: &Self::Config,
        _profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.main"),
                    operation_id: id("operation.main"),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
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
                initialization: StateInitialization::Zero,
                checkpoint: StateCheckpointCapability::Unsupported,
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
        Ok(ModelSemanticMetadata::CausalLanguage(
            ferrum_runtime::CausalLanguageMetadata {
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
            },
        ))
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
            binding: ResourcePresenceRequirement::Forbidden,
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
    let device_id: DeviceId = id(format!(
        "device.execution-fixture.{}",
        NEXT_DEVICE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.reference"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
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
            total_memory_bytes: 4096,
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
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?),
            Some(ProviderWorkspaceRequirement::new(
                32,
                16,
                ProviderWorkspaceScope::Plan,
                ProviderWorkspaceReusePolicy::Preserve,
                DynamicStorageRequirement::contiguous(),
            )?),
        ))
    }
}

impl OperationProvider<TestRuntime> for TestEstimator {
    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::Static)
    }

    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<EncodedDeviceOperation<()>, OperationFailure> {
        Ok(EncodedDeviceOperation::compute(()))
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
        None,
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
            Some(
                ResolvedWeightBinding::from_schema(family.weight_schema(), &id("weight.matrix"))
                    .unwrap(),
            ),
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

pub(crate) struct TestRuntime {
    descriptor: DeviceDescriptor,
}

pub(crate) type TestBuffer = BufferDescriptor;

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = ();
    type Command = ();
    type Fence = ();
    type Error = std::io::Error;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        Ok(BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        })
    }
    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.clone()
    }
    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(())
    }
    fn stream_state(&self, _: &Self::Stream) -> StreamState {
        StreamState::Ready
    }
    fn encode_copy(
        &self,
        _: &Self::Buffer,
        _: &Self::Buffer,
        _: CopyRegion,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn encode_upload(
        &self,
        _: &[u8],
        _: HostTransferLayout,
        _: &Self::Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn encode_zero(&self, _: &Self::Buffer, _: u64, _: u64) -> Result<(), Self::Error> {
        Ok(())
    }
    fn submit(
        &self,
        _: &mut Self::Stream,
        commands: DeviceCommandBatch<()>,
    ) -> Result<(), DefinitelyNotSubmitted<Self::Error>> {
        assert!(!commands.is_empty());
        Ok(())
    }
    fn query_fence(&self, _: &()) -> FenceQuery<Self::Error> {
        FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }
    fn wait_fence(
        &self,
        _: &(),
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }
    fn synchronize(&self, _: &mut Self::Stream) -> Result<(), Self::Error> {
        Ok(())
    }
    fn readback(
        &self,
        _: &mut Self::Stream,
        _: &Self::Buffer,
        _: CopyRegion,
        output: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![0; output.byte_len().unwrap() as usize])
    }
    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("capacity_fixture", error.to_string(), false)
    }
}

pub(crate) fn plan(
    catalog: &CapabilityCatalog,
    reusable: bool,
    maximum_active_sequences: u32,
) -> (ExecutionPlan, Option<ReusableExecutionBucketSpec>) {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare_with_profile(
            &serde_json::json!({"width": 4}),
            &NumericalProfileId::new("fixture.f32").unwrap(),
        )
        .unwrap();
    let bucket = reusable.then(|| {
        ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("execution.capacity-fixture").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap()
    });
    let reusable = bucket
        .as_ref()
        .map(|bucket| ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap());
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.capacity-fixture",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            checkpoint_capacity: None,
            capacity_bytes: 4096,
            reserve_bytes: 0,
            maximum_active_sequences,
            dynamic_storage_profile_order: vec![contiguous_storage_profile()],
        },
        AdmissionPolicy {
            maximum_queue_depth: 128,
            maximum_scheduled_tokens: 4096,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: Default::default(),
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
        reusable,
    )
    .unwrap();
    let registry = operation_registry(catalog);
    let resolution = node_resolution(&family, catalog, &policy, &registry.planning());
    (
        ExecutionPlan::build(
            PlanBuildRequest::new(&family, catalog, &policy, vec![resolution]).unwrap(),
        )
        .unwrap(),
        bucket,
    )
}

pub(crate) fn runtime(plan: &ExecutionPlan) -> Arc<TestRuntime> {
    Arc::new(TestRuntime {
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
    })
}

pub(crate) fn provision(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
    suffix: &str,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    provision_inner(plan, runtime, suffix, false)
}

pub(crate) fn provision_competing(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    provision_inner(plan, runtime, "pressure.competing", true)
}

fn provision_inner(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
    suffix: &str,
    fill_device: bool,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    let StaticProvisioning::Required(permit) = plan
        .provision_static(runtime.clone(), id(format!("request.provision.{suffix}")))
        .unwrap()
        .into_provisioning()
    else {
        panic!("fixture declares static weights")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id(format!("run.{suffix}")),
        id(format!("transaction.{suffix}")),
    );
    let transaction = ResourceTransaction::begin(
        RuntimeResourceDriver::new(runtime).unwrap(),
        identity,
        permit,
    )
    .unwrap()
    .reserve()
    .unwrap();
    let transaction = match transaction.commit() {
        Ok(transaction) => transaction,
        Err(ResourceCommitTransitionError::Recoverable(failure)) => {
            panic!("fixture commit failed: {:?}", failure.failure())
        }
        Err(ResourceCommitTransitionError::Poisoned(failure)) => {
            panic!("fixture commit poisoned: {:?}", failure.failure())
        }
    };
    let pools = transaction
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool in pools {
        transaction
            .maintenance_controller()
            .initialize_pool(&pool)
            .unwrap();
    }
    if fill_device {
        let before = transaction.maintenance_controller().status().unwrap();
        let pool = before
            .pools()
            .iter()
            .find(|pool| pool.contract().compatibility().usage() == BufferUsage::Scratch)
            .unwrap();
        let available =
            plan.payload().memory().device_capacity_bytes() - before.process_claimed_bytes();
        let growth = available / 16 * 16;
        assert!(growth > 0);
        // The competing plan has 128 declared 64-byte scratch instances, so
        // its pool ceiling reaches the real device budget rather than an
        // incidental smaller activation pool's theoretical ceiling.
        assert!(
            pool.resident_bytes() + growth
                <= pool.contract().provisioning().maximum_resident_bytes()
        );
        transaction
            .maintenance_controller()
            .grow_pool(pool.pool_id(), growth)
            .unwrap();
    }
    transaction
        .into_plan_runtime()
        .unwrap_or_else(|failure| panic!("runtime handoff: {}", failure.error()))
}

fn work() -> ResourceWorkShape {
    ResourceWorkShape::single(span()).unwrap()
}
fn span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[1], 0..1).unwrap()
}

pub(crate) fn sequence(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    suffix: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let binding = root.trusted_runtime_binding().unwrap();
    let request = RequestResourceAdmissionRequest::new(
        work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut admitted = None;
    for _ in 0..3 {
        match binding
            .try_admit_request(
                request.clone(),
                id(format!("run.{suffix}")),
                id(format!("request.{suffix}")),
            )
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(value) => {
                admitted = Some(value);
                break;
            }
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("fixture request admission rejected"),
        }
    }
    let request = admitted.expect("fixture request backing did not converge");
    let admission = SequenceResourceAdmissionRequest::new(
        work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for _ in 0..3 {
        match request.try_admit_sequence(admission.clone()).unwrap() {
            SequenceResourceAdmissionDecision::Admitted(value) => return value,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("fixture sequence admission rejected"),
        }
    }
    panic!("fixture sequence backing did not converge")
}

pub(crate) fn step(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let mut request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if let Some(bucket) = bucket {
        request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
    }
    for _ in 0..3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(value) => return value,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("fixture step admission rejected"),
        }
    }
    panic!("fixture step backing did not converge")
}

pub(crate) fn try_wave(
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> StepSubmissionWaveAdmissionDecision<TestRuntime> {
    step.try_prepare_full_plan_submission_wave(
        step.shared_all_invocation_work_shape(&[span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

pub(crate) fn wave(
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> PreparedStepSubmissionWave<TestRuntime> {
    for _ in 0..3 {
        match try_wave(step) {
            StepSubmissionWaveAdmissionDecision::Prepared(value) => return value,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("fixture submission wave admission rejected"),
        }
    }
    panic!("fixture submission wave backing did not converge")
}

pub(crate) fn close(root: Arc<PlanRuntimeResources<TestRuntime>>) {
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("runtime retained {strong_count} references")
        }
        Err(failure) => panic!("runtime close failed: {:?}", failure.failure()),
    }
}
