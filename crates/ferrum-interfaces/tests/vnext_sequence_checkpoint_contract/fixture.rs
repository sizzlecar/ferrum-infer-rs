use super::vnext_core_contract::*;

#[path = "fixture/family.rs"]
mod family;
use family::Family;

pub struct Spec {
    /// Optional isolated runtime account for resource tests sharing this fixture.
    pub device_id: Option<DeviceId>,
    pub family_id: ModelFamilyId,
    pub states: Vec<StateSpec>,
    pub declare_inputs: bool,
    pub conditioning: bool,
    pub output_only_input: Option<ProgramValueId>,
    pub output_only_feeds_state: bool,
    pub output_only_unknown_operation: bool,
    pub declare_provider: bool,
    pub declare_ports: bool,
    pub numerics: CheckpointPartitionNumerics,
    pub dependency: CheckpointInputDependency,
    pub boundaries: CheckpointBoundaryConstraint,
    pub profile: DynamicStorageProfile,
    pub port_profile: DynamicStorageProfile,
    pub hidden_persistent: bool,
    pub unselected_support: bool,
    pub read_only_state: bool,
    pub locations: Vec<(ResourceId, u64)>,
    pub layouts: Vec<ProviderCheckpointStateLayout>,
    pub checkpoint_capacity: Option<CheckpointCapacityPolicy>,
}

impl Default for Spec {
    fn default() -> Self {
        let states = [
            StateCheckpointContents::PrefixPositions,
            StateCheckpointContents::BoundaryValue,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, contents)| StateSpec {
            id: id(format!("state.{index}")),
            value_id: id(format!("value.state.{index}")),
            tensor: ProgramTensorSpec {
                dimensions: vec![4],
                element_type: ElementType::U8,
                layout: ResolvedTensorLayout::Contiguous,
            },
            lifetime: StateLifetime::Sequence,
            capacity_demand: if contents == StateCheckpointContents::PrefixPositions {
                StateCapacityDemand::TokenScaled {
                    bytes_per_token: 4,
                    maximum_tokens: 64,
                }
            } else {
                StateCapacityDemand::FixedPerScope
            },
            initialization: StateInitialization::Zero,
            checkpoint: StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
                contents,
                CheckpointInputDependency::ExactTokenPrefix,
            )),
        })
        .collect();
        Self {
            device_id: None,
            family_id: id("family.checkpoint-fixture"),
            states,
            declare_inputs: true,
            conditioning: false,
            output_only_input: None,
            output_only_feeds_state: false,
            output_only_unknown_operation: false,
            declare_provider: true,
            declare_ports: true,
            numerics: CheckpointPartitionNumerics::BitwiseEquivalent,
            dependency: CheckpointInputDependency::ExactTokenPrefix,
            boundaries: CheckpointBoundaryConstraint::any_positive(),
            profile: contiguous_storage_profile(),
            port_profile: contiguous_storage_profile(),
            hidden_persistent: false,
            unselected_support: false,
            read_only_state: false,
            locations: vec![(id("resource.state.0"), 0), (id("resource.state.1"), 0)],
            layouts: vec![
                ProviderCheckpointStateLayout::TokenMajorPrefix,
                ProviderCheckpointStateLayout::ContiguousBoundaryValue,
            ],
            checkpoint_capacity: None,
        }
    }
}

pub struct Fixture {
    pub family: PreparedModelFamily,
    pub catalog: CapabilityCatalog,
    pub policy: ResolvedRuntimePolicy,
    _registry: OperationRuntimeRegistry<PlanningTestRuntime>,
    resolutions: Vec<PlanNodeResolution>,
    pub plan: ExecutionPlan,
}

impl Fixture {
    pub fn build(spec: Spec) -> Result<Self, VNextError> {
        let family = TypedFamilyRegistration::new(Family {
            family_id: spec.family_id.clone(),
            states: spec.states.clone(),
            declare_inputs: spec.declare_inputs,
            conditioning: spec.conditioning,
            output_only_input: spec.output_only_input.clone(),
            output_only_feeds_state: spec.output_only_feeds_state,
            output_only_unknown_operation: spec.output_only_unknown_operation,
        })
        .prepare_fixture(&json!({"width": 4}))?;
        let operation = operation_for(&spec)?;
        let catalog = catalog_for(&spec, operation.clone())?;
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.checkpoint-fixture", ContractVersion::new(1, 0), SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy { checkpoint_capacity: spec.checkpoint_capacity, capacity_bytes: 8 << 20, reserve_bytes: 128, maximum_active_sequences: 3, dynamic_storage_profile_order: if spec.profile == contiguous_storage_profile() { vec![spec.profile] } else { vec![spec.profile, contiguous_storage_profile()] } },
            serde_json::from_value(json!({"maximum_queue_depth":8,"maximum_scheduled_tokens":4096,"sequence_fit_policy":"immediate_only","allow_defer":true,"cancellation_check_interval_steps":1})).unwrap(),
            ferrum_types::AttentionExecutionPolicy::Portable, ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay, None,
        )?;
        let mut operations = vec![operation];
        if spec.output_only_input.is_some() {
            operations.push(output_operation());
        }
        let registry = OperationRuntimeRegistry::new(
            operations
                .into_iter()
                .map(|descriptor| {
                    Box::new(TestOperationContract {
                        descriptor,
                        calls: Arc::new(AtomicUsize::new(0)),
                        reject_signature: false,
                    }) as Box<dyn OperationContract>
                })
                .collect(),
            catalog
                .providers()
                .values()
                .flatten()
                .cloned()
                .map(|descriptor| {
                    Box::new(Provider {
                        descriptor,
                        hidden_persistent: spec.hidden_persistent,
                    }) as Box<dyn OperationProvider<PlanningTestRuntime>>
                })
                .collect(),
        )?;
        let resolution = PlanNodeResolution::resolve(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            id("node.main"),
            values_for(&spec)?,
            BTreeSet::new(),
            Some(id("provider.selected")),
        )?;
        let mut resolutions = vec![resolution.clone()];
        if let Some(input) = &spec.output_only_input {
            resolutions.push(PlanNodeResolution::resolve(
                &family,
                &catalog,
                &policy,
                &registry.planning(),
                id("node.output"),
                vec![
                    binding(
                        "value.output",
                        ResolvedValueRole::Input,
                        0,
                        ElementType::F32,
                        TensorAccess::Read,
                        BufferUsage::Activations,
                        "resource.output".to_owned(),
                    ),
                    binding(
                        input.as_str(),
                        ResolvedValueRole::Input,
                        1,
                        ElementType::F32,
                        TensorAccess::Read,
                        BufferUsage::Activations,
                        "resource.selection".to_owned(),
                    ),
                    binding(
                        "value.final",
                        ResolvedValueRole::Output,
                        0,
                        ElementType::F32,
                        TensorAccess::Write,
                        BufferUsage::Activations,
                        "resource.final".to_owned(),
                    ),
                ],
                BTreeSet::new(),
                Some(id("provider.output")),
            )?);
        }
        let plan = ExecutionPlan::build(PlanBuildRequest::new(
            &family,
            &catalog,
            &policy,
            resolutions.clone(),
        )?)?;
        Ok(Self {
            family,
            catalog,
            policy,
            _registry: registry,
            resolutions,
            plan,
        })
    }

    pub fn layout(&self) -> &SequenceCheckpointLayout {
        match self.plan.sequence_checkpoint_capability() {
            SequenceCheckpointCapability::Enabled(layout) => layout,
            SequenceCheckpointCapability::Unsupported(reasons) => {
                panic!("unexpected unsupported layout: {reasons:?}")
            }
        }
    }

    pub fn reasons(&self) -> &[SequenceCheckpointUnsupportedReason] {
        match self.plan.sequence_checkpoint_capability() {
            SequenceCheckpointCapability::Unsupported(reasons) => reasons,
            SequenceCheckpointCapability::Enabled(_) => panic!("unexpected checkpoint support"),
        }
    }

    pub fn revalidate(&self, wire: &[u8]) -> Result<ExecutionPlan, VNextError> {
        ExecutionPlan::from_json_validated(
            wire,
            &self.family,
            &self.catalog,
            &self.policy,
            self.resolutions.clone(),
        )
    }
}

fn operation_for(spec: &Spec) -> Result<OperationDescriptor, VNextError> {
    let token = |dtype, access| {
        TensorContract::new(
            vec![DimensionConstraint::Symbol("tokens".to_owned())],
            BTreeSet::from([dtype]),
            vec![LayoutConstraint::Contiguous],
            access,
            AliasPolicy::NoAlias,
        )
    };
    let mut operation = operation();
    operation.inputs = vec![
        token(ElementType::U32, TensorAccess::Read)?,
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
    ];
    for state in &spec.states {
        operation.inputs.push(TensorContract::new(
            state
                .tensor
                .dimensions
                .iter()
                .copied()
                .map(DimensionConstraint::Exact)
                .collect(),
            BTreeSet::from([state.tensor.element_type]),
            vec![LayoutConstraint::Contiguous],
            if spec.read_only_state {
                TensorAccess::Read
            } else {
                TensorAccess::ReadWrite
            },
            AliasPolicy::NoAlias,
        )?);
    }
    if spec.conditioning {
        operation.inputs.push(tensor_contract(
            ElementType::F32,
            TensorAccess::Read,
            AliasPolicy::NoAlias,
        ));
    }
    if spec.output_only_input.is_some() && spec.output_only_feeds_state {
        operation.inputs.push(tensor_contract(
            ElementType::F32,
            TensorAccess::Read,
            AliasPolicy::NoAlias,
        ));
    }
    operation.outputs = vec![token(ElementType::F32, TensorAccess::Write)?];
    operation.resources.scratch = ResourcePresenceRequirement::Forbidden;
    operation.resources.persistent = if spec.hidden_persistent {
        ResourcePresenceRequirement::Required
    } else {
        ResourcePresenceRequirement::Forbidden
    };
    operation.profile_phase = ProfilePhase::Forward;
    Ok(operation)
}

fn catalog_for(
    spec: &Spec,
    operation: OperationDescriptor,
) -> Result<CapabilityCatalog, VNextError> {
    let profiles = vec![contiguous_storage_profile(), paged_storage_profile(65536)];
    let make_provider = |name: &str, declare: bool| -> Result<_, VNextError> {
        let bindings = storage_bindings(
            &operation,
            DynamicStorageRequirement::new(profiles.clone())?,
        );
        let mut provider = OperationProviderDescriptor::new(
            id(name),
            operation.id.clone(),
            operation.fingerprint()?,
            sha('f'),
            ProviderExecutionSemantics::bitwise_eager_and_replay(),
            ContractVersion::new(1, 0),
            spec.device_id
                .clone()
                .unwrap_or_else(|| id("device.reference.0")),
            BTreeSet::from([id("capability.compute")]),
            BTreeSet::from([id("weight-format.dense")]),
            BTreeSet::new(),
            bindings,
            "resource-estimator.checkpoint",
            ContractVersion::new(1, 0),
            sha('e'),
        )?;
        if declare {
            let ports = if spec.declare_ports {
                spec.states
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        ProviderCheckpointStatePort::new(
                            ResolvedValueRole::Input,
                            u32::try_from(index + 2).unwrap(),
                            spec.port_profile,
                            spec.layouts[index],
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
            provider = provider.with_checkpoint_capability(
                ProviderCheckpointCapability::CompletedBoundary(
                    ProviderCheckpointContract::new(
                        spec.dependency,
                        spec.boundaries,
                        spec.numerics,
                    )
                    .with_state_ports(ports)?,
                ),
            );
        }
        Ok(provider)
    };
    let mut providers = vec![make_provider("provider.selected", spec.declare_provider)?];
    if spec.unselected_support {
        providers.push(make_provider("provider.alternative", true)?);
    }
    let original = catalog();
    let mut device = original.device().clone();
    if let Some(id) = &spec.device_id {
        device.id = id.clone();
    }
    device.total_memory_bytes = 16 << 20;
    device.dynamic_storage_profiles = profiles.into_iter().collect();
    let mut operations = vec![operation.clone()];
    let mut provider_map = BTreeMap::from([(operation.id.clone(), providers)]);
    if spec.output_only_input.is_some() {
        let output = output_operation();
        let mut provider = OperationProviderDescriptor::new(
            id("provider.output"),
            output.id.clone(),
            output.fingerprint()?,
            sha('a'),
            ProviderExecutionSemantics::bitwise_eager_and_replay(),
            ContractVersion::new(1, 0),
            spec.device_id
                .clone()
                .unwrap_or_else(|| id("device.reference.0")),
            BTreeSet::from([id("capability.compute")]),
            BTreeSet::from([id("weight-format.dense")]),
            BTreeSet::new(),
            storage_bindings(&output, contiguous_storage_requirement()),
            "resource-estimator.checkpoint-output",
            ContractVersion::new(1, 0),
            sha('b'),
        )?;
        if spec.declare_provider {
            provider = provider.with_checkpoint_capability(
                ProviderCheckpointCapability::CompletedBoundary(ProviderCheckpointContract::new(
                    spec.dependency,
                    spec.boundaries,
                    spec.numerics,
                )),
            );
        }
        provider_map.insert(output.id.clone(), vec![provider]);
        operations.push(output);
    }
    CapabilityCatalog::new(
        device,
        operations,
        provider_map,
        original
            .engine_providers()
            .values()
            .map(|provider| {
                EngineProviderDescriptor::new(
                    provider.provider_id().clone(),
                    provider.contract_version(),
                    provider.implementation_fingerprint(),
                    spec.device_id
                        .clone()
                        .unwrap_or_else(|| provider.device_id().clone()),
                    provider.capabilities().clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
}

fn output_operation() -> OperationDescriptor {
    let mut output = operation();
    output.id = id("operation.output");
    output.inputs = vec![
        TensorContract::new(
            vec![DimensionConstraint::Symbol("tokens".to_owned())],
            BTreeSet::from([ElementType::F32]),
            vec![LayoutConstraint::Contiguous],
            TensorAccess::Read,
            AliasPolicy::NoAlias,
        )
        .unwrap(),
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
    ];
    output.outputs = vec![tensor_contract(
        ElementType::F32,
        TensorAccess::Write,
        AliasPolicy::NoAlias,
    )];
    output.resources.scratch = ResourcePresenceRequirement::Forbidden;
    output.resources.persistent = ResourcePresenceRequirement::Forbidden;
    output.profile_phase = ProfilePhase::Forward;
    output
}

fn values_for(spec: &Spec) -> Result<Vec<ResolvedValueBinding>, VNextError> {
    let original = resolved_values(0);
    let mut values = vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            ElementType::U32,
            TensorAccess::Read,
            BufferUsage::Activations,
            "resource.input".to_owned(),
        ),
        original[1].clone(),
    ];
    for (index, state) in spec.states.iter().enumerate() {
        let (resource_id, offset) = &spec.locations[index];
        let tensor = ResolvedTensorSpec::new(
            state.tensor.dimensions.clone(),
            state.tensor.element_type,
            state.tensor.layout.clone(),
        )?;
        values.push(ResolvedValueBinding::new(
            state.value_id.clone(),
            ResolvedValueRole::Input,
            u32::try_from(index + 2).unwrap(),
            tensor,
            if spec.read_only_state {
                TensorAccess::Read
            } else {
                TensorAccess::ReadWrite
            },
            AliasPolicy::NoAlias,
            BufferUsage::State,
            None,
            ResolvedValueStorage::single(
                resource_id.clone(),
                *offset,
                state.tensor.byte_len()?,
                state.tensor.element_type,
            )?,
        )?);
    }
    if spec.conditioning {
        values.push(binding(
            "value.conditioning",
            ResolvedValueRole::Input,
            u32::try_from(spec.states.len() + 2).unwrap(),
            ElementType::F32,
            TensorAccess::Read,
            BufferUsage::Activations,
            "resource.conditioning".to_owned(),
        ));
    }
    if let Some(input) = spec
        .output_only_input
        .as_ref()
        .filter(|_| spec.output_only_feeds_state)
    {
        values.push(binding(
            input.as_str(),
            ResolvedValueRole::Input,
            u32::try_from(spec.states.len() + 2 + usize::from(spec.conditioning)).unwrap(),
            ElementType::F32,
            TensorAccess::Read,
            BufferUsage::Activations,
            "resource.selection".to_owned(),
        ));
    }
    values.push(binding(
        "value.output",
        ResolvedValueRole::Output,
        0,
        ElementType::F32,
        TensorAccess::Write,
        BufferUsage::Activations,
        "resource.output".to_owned(),
    ));
    Ok(values)
}

struct Provider {
    descriptor: OperationProviderDescriptor,
    hidden_persistent: bool,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let persistent = self
            .hidden_persistent
            .then(|| {
                ProviderWorkspaceRequirement::new(
                    16,
                    16,
                    ProviderWorkspaceScope::Sequence,
                    ProviderWorkspaceReusePolicy::Preserve,
                    contiguous_storage_requirement(),
                )
            })
            .transpose()?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            persistent,
        ))
    }
}
impl OperationProvider<PlanningTestRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::Static)
    }
    fn encode_selected(
        &self,
        _: BatchedOperationInvocation<'_, BufferDescriptor>,
    ) -> Result<EncodedDeviceOperation<()>, OperationFailure> {
        Ok(EncodedDeviceOperation::compute(()))
    }
}
