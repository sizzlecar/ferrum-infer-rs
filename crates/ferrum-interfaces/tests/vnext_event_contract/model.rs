use super::*;

pub(crate) const MAX_EVENT_MAINTENANCE_ATTEMPTS: usize = 8;

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
    #[serde(default)]
    pub(crate) no_static: bool,
}

#[derive(Default)]
pub(crate) struct TestFamily;

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
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![id("value.input"), id("value.weight")],
                        outputs: vec![id("value.middle")],
                        attributes: BTreeMap::new(),
                    },
                    ProgramNode {
                        id: id("node.second"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
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

pub(crate) fn tensor_contract(access: TensorAccess) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(4)],
        BTreeSet::from([ElementType::F32]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

pub(crate) fn operation() -> OperationDescriptor {
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
                reason: "event fixture operation signature mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

pub(crate) fn policy() -> ResolvedRuntimePolicy {
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
    ResolvedTensorSpec::new(vec![4], ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
}

pub(crate) fn binding(
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

pub(crate) fn make_operation_registry(
    catalog: &CapabilityCatalog,
) -> OperationRuntimeRegistry<TestRuntime> {
    OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
        })],
        vec![Box::new(TestExecutionProvider::new(catalog))],
    )
    .unwrap()
}

pub(crate) fn execution_plan(
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    execution_plan_with_mode(suffix, operation_registry, false)
}

pub(crate) fn no_static_execution_plan(
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    execution_plan_with_mode(suffix, operation_registry, true)
}

pub(crate) fn execution_plan_with_mode(
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
                Some(
                    ResolvedWeightBinding::from_schema(
                        family.weight_schema(),
                        &id("weight.matrix"),
                    )
                    .unwrap(),
                ),
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
