pub(crate) use ferrum_interfaces::vnext::*;
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::{json, Value};
pub(crate) use sha2::{Digest, Sha256};
pub(crate) use std::collections::{BTreeMap, BTreeSet};
pub(crate) use std::fs;
pub(crate) use std::path::PathBuf;
pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};
pub(crate) use std::sync::Arc;

pub(crate) fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

pub(crate) fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

pub(crate) fn resource_work(sequence_token_counts: &[usize]) -> ResourceWorkShape {
    ResourceWorkShape::from_token_spans(
        sequence_token_counts
            .iter()
            .enumerate()
            .map(|(sequence, count)| {
                let tokens = (0..*count)
                    .map(|token| u32::try_from(sequence * 1024 + token).unwrap())
                    .collect::<Vec<_>>();
                TokenSpanWork::from_token_ids(&tokens, 0..tokens.len()).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

pub(crate) fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

pub(crate) fn contiguous_storage_requirement() -> DynamicStorageRequirement {
    DynamicStorageRequirement::contiguous()
}

pub(crate) fn storage_bindings(
    operation: &OperationDescriptor,
    requirement: DynamicStorageRequirement,
) -> Vec<ProviderStorageBindingRequirement> {
    operation
        .inputs
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal as u32,
                requirement.clone(),
            )
        })
        .chain(operation.outputs.iter().enumerate().map(|(ordinal, _)| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                ordinal as u32,
                requirement.clone(),
            )
        }))
        .collect()
}

pub(crate) fn contiguous_storage_bindings(
    operation: &OperationDescriptor,
) -> Vec<ProviderStorageBindingRequirement> {
    storage_bindings(operation, contiguous_storage_requirement())
}

pub(crate) fn paged_storage_profile(block_bytes: u64) -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena { block_bytes },
        DynamicStorageView::PagedRegions { block_bytes },
    )
    .unwrap()
}

pub(crate) fn paged_storage_requirement(block_bytes: u64) -> DynamicStorageRequirement {
    DynamicStorageRequirement::new(vec![paged_storage_profile(block_bytes)]).unwrap()
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
        FAMILY.get_or_init(|| id("family.synthetic"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.synthetic"), id("metadata.synthetic.alias")])
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
        Ok(id("metadata.synthetic"))
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
                reason: "test family requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.dense"),
            layout_id: id("weight-layout.dense"),
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
                sha256: sha('c'),
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

pub(crate) struct OrderedSchemaFamily {
    pub(crate) reverse: bool,
}

impl ModelFamilyProvider for OrderedSchemaFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        let mut schema = TestFamily.weight_schema(config)?;
        schema.components[0].external_names = if self.reverse {
            vec!["weight.z".to_owned(), "weight.a".to_owned()]
        } else {
            vec!["weight.a".to_owned(), "weight.z".to_owned()]
        };
        for suffix in ["a", "b"] {
            schema.components.push(WeightComponentSpec {
                id: id(format!("weight.component.optional.{suffix}")),
                role: WeightComponentRole::Values,
                external_names: if self.reverse {
                    vec![
                        format!("optional.{suffix}.z"),
                        format!("optional.{suffix}.a"),
                    ]
                } else {
                    vec![
                        format!("optional.{suffix}.a"),
                        format!("optional.{suffix}.z"),
                    ]
                },
                dimensions: vec![config.width / 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: false,
            });
        }
        let mut parts = vec![
            CompositeWeightPart {
                layout: Box::new(PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.optional.a"),
                }),
                logical_offsets: vec![0],
                extents: vec![config.width / 2],
            },
            CompositeWeightPart {
                layout: Box::new(PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.optional.b"),
                }),
                logical_offsets: vec![config.width / 2],
                extents: vec![config.width / 2],
            },
        ];
        if self.reverse {
            parts.reverse();
        }
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.optional"),
            dimensions: vec![config.width],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Composite { parts },
            required: false,
        });
        if self.reverse {
            schema.components.reverse();
            schema.tensors.reverse();
        }
        Ok(schema)
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        TestFamily.semantic_program(config)
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
}

pub(crate) struct FixedSchemaFamily {
    pub(crate) schema: WeightSchema,
}

impl ModelFamilyProvider for FixedSchemaFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(self.schema.clone())
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        TestFamily.semantic_program(config)
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
}

pub(crate) struct TestRegistry {
    pub(crate) registration: TypedFamilyRegistration<TestFamily>,
}

impl TestRegistry {
    pub(crate) fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }

    pub(crate) fn prepare(&self) -> PreparedModelFamily {
        self.registration.prepare(&json!({"width": 4})).unwrap()
    }
}

impl ModelFamilyRegistry for TestRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

#[derive(Default)]
pub(crate) struct SequentialScratchFamily;

impl ModelFamilyProvider for SequentialScratchFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.sequential-scratch"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.sequential-scratch")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.sequential-scratch"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        TestFamily.weight_schema(config)
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.sequential".to_owned(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.first"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        inputs: vec![id("value.input"), id("value.weight"), id("value.state")],
                        outputs: vec![id("value.intermediate")],
                        attributes: BTreeMap::new(),
                    },
                    ProgramNode {
                        id: id("node.second"),
                        operation_id: id("operation.main"),
                        required_version: ContractVersion::new(1, 0),
                        inputs: vec![
                            id("value.intermediate"),
                            id("value.weight"),
                            id("value.state"),
                        ],
                        outputs: vec![id("value.output")],
                        attributes: BTreeMap::new(),
                    },
                ],
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
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
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
    catalog_with_memory(1 << 20)
}

pub(crate) fn catalog_with_memory(total_memory_bytes: u64) -> CapabilityCatalog {
    let operation = operation();
    let device_id: DeviceId = id("device.reference.0");
    let capabilities: BTreeSet<CapabilityId> = BTreeSet::from([id("capability.compute")]);
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.reference"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('f'),
        ContractVersion::new(1, 0),
        device_id.clone(),
        capabilities.clone(),
        BTreeSet::from([id("weight-format.dense")]),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation),
        "resource-estimator.reference",
        ContractVersion::new(1, 0),
        sha('e'),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.reference"),
        ContractVersion::new(1, 0),
        sha('8'),
        device_id.clone(),
        capabilities.clone(),
    )
    .unwrap();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id,
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes,
            runtime_implementation_fingerprint: sha('d'),
            capabilities,
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        vec![operation.clone()],
        BTreeMap::from([(operation.id.clone(), vec![provider])]),
        vec![engine],
    )
    .unwrap()
}

pub(crate) fn catalog_from_operations(
    operations: Vec<OperationDescriptor>,
) -> Result<CapabilityCatalog, VNextError> {
    let device_id: DeviceId = id("device.reference.0");
    let capabilities: BTreeSet<CapabilityId> = BTreeSet::from([id("capability.compute")]);
    let providers = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| {
            Ok((
                operation.id.clone(),
                vec![OperationProviderDescriptor::new(
                    id(format!("provider.operation.oracle.{index}")),
                    operation.id.clone(),
                    operation.fingerprint()?,
                    sha('f'),
                    ContractVersion::new(1, 0),
                    device_id.clone(),
                    capabilities.clone(),
                    BTreeSet::from([id("weight-format.dense")]),
                    BTreeSet::new(),
                    contiguous_storage_bindings(operation),
                    format!("resource-estimator.oracle.{index}"),
                    ContractVersion::new(1, 0),
                    sha('e'),
                )?],
            ))
        })
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
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
        operations,
        providers,
        vec![EngineProviderDescriptor::new(
            id("provider.engine.oracle"),
            ContractVersion::new(1, 0),
            sha('8'),
            device_id,
            capabilities,
        )?],
    )
}

pub(crate) fn catalog_with_secondary_provider() -> CapabilityCatalog {
    catalog_with_secondary_provider_storage(
        contiguous_storage_requirement(),
        contiguous_storage_requirement(),
    )
}

pub(crate) fn catalog_with_secondary_provider_storage(
    reference_storage: DynamicStorageRequirement,
    secondary_storage: DynamicStorageRequirement,
) -> CapabilityCatalog {
    let operation = operation();
    let device_id: DeviceId = id("device.reference.0");
    let capabilities: BTreeSet<CapabilityId> = BTreeSet::from([id("capability.compute")]);
    let provider = |provider_id: &str,
                    provider_implementation: char,
                    estimator_id: &str,
                    estimator_implementation: char,
                    storage: DynamicStorageRequirement| {
        OperationProviderDescriptor::new(
            id(provider_id),
            operation.id.clone(),
            operation.fingerprint().unwrap(),
            sha(provider_implementation),
            ContractVersion::new(1, 0),
            device_id.clone(),
            capabilities.clone(),
            BTreeSet::from([id("weight-format.dense")]),
            BTreeSet::new(),
            storage_bindings(&operation, storage),
            estimator_id,
            ContractVersion::new(1, 0),
            sha(estimator_implementation),
        )
        .unwrap()
    };
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
        BTreeMap::from([(
            operation.id.clone(),
            vec![
                provider(
                    "provider.operation.reference",
                    '6',
                    "resource-estimator.reference",
                    'e',
                    reference_storage,
                ),
                provider(
                    "provider.operation.secondary",
                    '7',
                    "resource-estimator.secondary",
                    'f',
                    secondary_storage,
                ),
            ],
        )]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.reference"),
            ContractVersion::new(1, 0),
            sha('8'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

pub(crate) struct TestOperationContract {
    pub(crate) descriptor: OperationDescriptor,
    pub(crate) calls: Arc<AtomicUsize>,
    pub(crate) reject_signature: bool,
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.reject_signature
            || inputs != self.descriptor.inputs
            || outputs != self.descriptor.outputs
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "typed operation signature rejected".to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EstimateBehavior {
    Correct,
    WrongEstimatorId,
    WrongEstimatorVersion,
    WrongImplementation,
    WrongInput,
    InvalidAlignment,
    MissingScratch,
}

pub(crate) struct TestEstimator {
    pub(crate) descriptor: OperationProviderDescriptor,
    pub(crate) calls: Arc<AtomicUsize>,
    pub(crate) scratch_bytes: u64,
    pub(crate) persistent_bytes: u64,
    pub(crate) behavior: EstimateBehavior,
}

pub(crate) struct PlanningTestRuntime;

impl DeviceRuntime for PlanningTestRuntime {
    type Buffer = BufferDescriptor;
    type Stream = ();
    type Command = ();
    type Fence = ();
    type Error = std::io::Error;

    fn descriptor(&self) -> &DeviceDescriptor {
        static DESCRIPTOR: std::sync::OnceLock<DeviceDescriptor> = std::sync::OnceLock::new();
        DESCRIPTOR.get_or_init(|| DeviceDescriptor {
            id: id("device.reference.0"),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: sha('d'),
            capabilities: BTreeSet::from([id("capability.compute")]),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        })
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
        FenceQuery::Terminal(DeviceTerminal::Succeeded)
    }

    fn wait_fence(
        &self,
        _fence: &Self::Fence,
    ) -> Result<DeviceTerminal<Self::Error>, FenceIndeterminate<Self::Error>> {
        Ok(DeviceTerminal::Succeeded)
    }

    fn synchronize(&self, _stream: &mut Self::Stream) -> Result<(), Self::Error> {
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
        DeviceErrorReport::new("planning_test_runtime", error.to_string(), false)
    }
}

impl OperationResourceEstimator for TestEstimator {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let estimator_id = if self.behavior == EstimateBehavior::WrongEstimatorId {
            "resource-estimator.wrong"
        } else {
            self.descriptor.resource_estimator_id()
        };
        let estimator_version = if self.behavior == EstimateBehavior::WrongEstimatorVersion {
            ContractVersion::new(2, 0)
        } else {
            self.descriptor.resource_estimator_version()
        };
        let implementation = if self.behavior == EstimateBehavior::WrongImplementation {
            sha('0')
        } else {
            self.descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned()
        };
        let input = if self.behavior == EstimateBehavior::WrongInput {
            sha('1')
        } else {
            request.input_fingerprint().to_owned()
        };
        let scratch = (self.behavior != EstimateBehavior::MissingScratch).then(|| {
            ProviderWorkspaceRequirement::new(
                self.scratch_bytes,
                16,
                ProviderWorkspaceScope::Invocation,
                contiguous_storage_requirement(),
            )
            .unwrap()
        });
        Ok(OperationResourceEstimate::new(
            estimator_id,
            estimator_version,
            implementation,
            input,
            if self.behavior == EstimateBehavior::InvalidAlignment {
                3
            } else {
                16
            },
            scratch,
            Some(
                ProviderWorkspaceRequirement::new(
                    self.persistent_bytes,
                    16,
                    ProviderWorkspaceScope::Plan,
                    contiguous_storage_requirement(),
                )
                .unwrap(),
            ),
        ))
    }
}

impl OperationProvider<PlanningTestRuntime> for TestEstimator {
    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, BufferDescriptor>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

pub(crate) struct SequentialScratchEstimator {
    pub(crate) descriptor: OperationProviderDescriptor,
}

impl OperationResourceEstimator for SequentialScratchEstimator {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let scratch_bytes = match request.node_id().as_str() {
            "node.first" => 64,
            "node.second" => 96,
            other => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("unexpected sequential node `{other}`"),
                })
            }
        };
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            Some(ProviderWorkspaceRequirement::new(
                scratch_bytes,
                16,
                ProviderWorkspaceScope::Invocation,
                contiguous_storage_requirement(),
            )?),
            Some(ProviderWorkspaceRequirement::new(
                32,
                16,
                ProviderWorkspaceScope::Plan,
                contiguous_storage_requirement(),
            )?),
        ))
    }
}

impl OperationProvider<PlanningTestRuntime> for SequentialScratchEstimator {
    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, BufferDescriptor>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

pub(crate) struct TestPlanningEntries {
    pub(crate) contracts: Vec<TestOperationContract>,
    pub(crate) estimators: Vec<TestEstimator>,
    pub(crate) contract_calls: Arc<AtomicUsize>,
    pub(crate) estimator_calls: Arc<AtomicUsize>,
}

impl TestPlanningEntries {
    pub(crate) fn new(
        catalog: &CapabilityCatalog,
        scratch_bytes: u64,
        persistent_bytes: u64,
        behavior: EstimateBehavior,
    ) -> Self {
        let contract_calls = Arc::new(AtomicUsize::new(0));
        let estimator_calls = Arc::new(AtomicUsize::new(0));
        Self {
            contracts: catalog
                .operations()
                .values()
                .cloned()
                .map(|descriptor| TestOperationContract {
                    descriptor,
                    calls: contract_calls.clone(),
                    reject_signature: false,
                })
                .collect(),
            estimators: catalog
                .providers()
                .values()
                .flatten()
                .cloned()
                .map(|descriptor| TestEstimator {
                    descriptor,
                    calls: estimator_calls.clone(),
                    scratch_bytes,
                    persistent_bytes,
                    behavior,
                })
                .collect(),
            contract_calls,
            estimator_calls,
        }
    }

    pub(crate) fn build(self) -> Result<TestPlanningRegistry, VNextError> {
        let registry = OperationRuntimeRegistry::new(
            self.contracts
                .into_iter()
                .map(|contract| Box::new(contract) as Box<dyn OperationContract>)
                .collect(),
            self.estimators
                .into_iter()
                .map(|provider| {
                    Box::new(provider) as Box<dyn OperationProvider<PlanningTestRuntime>>
                })
                .collect(),
        )?;
        Ok(TestPlanningRegistry {
            registry,
            contract_calls: self.contract_calls,
            estimator_calls: self.estimator_calls,
        })
    }
}

pub(crate) struct TestPlanningRegistry {
    pub(crate) registry: OperationRuntimeRegistry<PlanningTestRuntime>,
    pub(crate) contract_calls: Arc<AtomicUsize>,
    pub(crate) estimator_calls: Arc<AtomicUsize>,
}

impl TestPlanningRegistry {
    pub(crate) fn new(
        catalog: &CapabilityCatalog,
        scratch_bytes: u64,
        persistent_bytes: u64,
        behavior: EstimateBehavior,
    ) -> Self {
        TestPlanningEntries::new(catalog, scratch_bytes, persistent_bytes, behavior)
            .build()
            .unwrap()
    }

    pub(crate) fn planning(&self) -> OperationPlanningHandle<'_> {
        self.registry.planning()
    }
}

pub(crate) fn policy(capacity_bytes: u64) -> ResolvedRuntimePolicy {
    policy_with(capacity_bytes, 128, 3).unwrap()
}

pub(crate) fn policy_with(
    capacity_bytes: u64,
    reserve_bytes: u64,
    maximum_active_sequences: u32,
) -> Result<ResolvedRuntimePolicy, VNextError> {
    ResolvedRuntimePolicy::new(
        "runtime-policy.test",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes,
            reserve_bytes,
            maximum_active_sequences,
            dynamic_storage_profile_order: vec![contiguous_storage_profile()],
        },
        serde_json::from_value(json!({
            "maximum_queue_depth": 8,
            "allow_defer": true,
            "cancellation_check_interval_steps": 1
        }))
        .unwrap(),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AdversarialRuntimePolicy {
    pub(crate) maximum_active_sequences: u32,
    pub(crate) dynamic_storage_profile_order: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for AdversarialRuntimePolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }

    fn memory_capacity_bytes(&self) -> u64 {
        1 << 20
    }

    fn memory_reserve_bytes(&self) -> u64 {
        128
    }

    fn maximum_active_sequences(&self) -> u32 {
        self.maximum_active_sequences
    }

    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.dynamic_storage_profile_order
    }

    fn validate(&self) -> Result<(), VNextError> {
        Ok(())
    }
}

pub(crate) fn resolved_tensor(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

pub(crate) fn binding(
    value_id: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    element_type: ElementType,
    access: TensorAccess,
    usage: BufferUsage,
    resource_id: String,
) -> ResolvedValueBinding {
    let length = 4 * element_type.size_bytes();
    ResolvedValueBinding::new(
        id(value_id),
        role,
        ordinal,
        resolved_tensor(element_type),
        access,
        AliasPolicy::NoAlias,
        usage,
        ResolvedValueStorage::single(id(resource_id), 0, length, element_type).unwrap(),
    )
    .unwrap()
}

pub(crate) fn resolved_values(variant: usize) -> Vec<ResolvedValueBinding> {
    let weight_storage = ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
        Some(id("weight.component")),
        id(format!("resource.weight.{variant}")),
        0,
        16,
        ElementType::F32,
    )
    .unwrap()])
    .unwrap();
    vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            ElementType::F32,
            TensorAccess::Read,
            BufferUsage::Activations,
            format!("resource.input.{variant}"),
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
            format!("resource.state.{variant}"),
        ),
        binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            ElementType::F32,
            TensorAccess::Write,
            BufferUsage::Activations,
            format!("resource.output.{variant}"),
        ),
    ]
}

pub(crate) fn sequential_resolved_values(
    input_value: &str,
    input_resource: &str,
    output_value: &str,
    output_resource: &str,
) -> Vec<ResolvedValueBinding> {
    let weight_storage = ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
        Some(id("weight.component")),
        id("resource.sequential.weight"),
        0,
        16,
        ElementType::F32,
    )
    .unwrap()])
    .unwrap();
    vec![
        binding(
            input_value,
            ResolvedValueRole::Input,
            0,
            ElementType::F32,
            TensorAccess::Read,
            BufferUsage::Activations,
            input_resource.to_owned(),
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
            "resource.sequential.state".to_owned(),
        ),
        binding(
            output_value,
            ResolvedValueRole::Output,
            0,
            ElementType::F32,
            TensorAccess::Write,
            BufferUsage::Activations,
            output_resource.to_owned(),
        ),
    ]
}

pub(crate) fn try_node_resolution_with_registry(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &ResolvedRuntimePolicy,
    variant: usize,
    planning: &TestPlanningRegistry,
    preferred_provider: Option<&str>,
) -> Result<PlanNodeResolution, VNextError> {
    let planning = planning.planning();
    PlanNodeResolution::resolve(
        family,
        catalog,
        policy,
        &planning,
        id("node.main"),
        resolved_values(variant),
        BTreeSet::new(),
        preferred_provider.map(id),
    )
}

pub(crate) fn node_resolution_with_registry(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &ResolvedRuntimePolicy,
    variant: usize,
    planning: &TestPlanningRegistry,
    preferred_provider: Option<&str>,
) -> PlanNodeResolution {
    try_node_resolution_with_registry(
        family,
        catalog,
        policy,
        variant,
        planning,
        preferred_provider,
    )
    .unwrap()
}

pub(crate) fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &ResolvedRuntimePolicy,
    variant: usize,
    planning: &TestPlanningRegistry,
) -> PlanNodeResolution {
    node_resolution_with_registry(family, catalog, policy, variant, planning, None)
}

pub(crate) struct PlanFixture {
    pub(crate) registry: TestRegistry,
    pub(crate) family: PreparedModelFamily,
    pub(crate) catalog: CapabilityCatalog,
    pub(crate) policy: ResolvedRuntimePolicy,
    pub(crate) planning: TestPlanningRegistry,
    pub(crate) node_resolutions: Vec<PlanNodeResolution>,
    pub(crate) plan: ExecutionPlan,
}

pub(crate) fn plan_fixture(variant: usize) -> PlanFixture {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let node_resolutions = vec![node_resolution(
        &family, &catalog, &policy, variant, &planning,
    )];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, node_resolutions.clone()).unwrap(),
    )
    .unwrap();
    PlanFixture {
        registry,
        family,
        catalog,
        policy,
        planning,
        node_resolutions,
        plan,
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GraphConfig {
    pub(crate) scenario: String,
}

pub(crate) struct GraphFamily;

impl ModelFamilyProvider for GraphFamily {
    type Config = GraphConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.execution-graph"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.execution-graph")])
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
        Ok(id("metadata.execution-graph"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: GraphConfig = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "scenario".to_owned(),
                reason: error.to_string(),
            }
        })?;
        if !matches!(
            config.scenario.as_str(),
            "alias" | "alias_late_consumer" | "state_chain" | "state_read_only"
        ) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "scenario".to_owned(),
                reason: "unknown execution graph scenario".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.execution-graph"),
            layout_id: id("weight-layout.execution-graph"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component"),
                role: WeightComponentRole::Values,
                external_names: vec!["weight.bin".to_owned()],
                dimensions: vec![4],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![4],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component"),
                },
                required: true,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let node = |node_id: &str, operation_id: &str, output: &str, state: bool| ProgramNode {
            id: id(node_id),
            operation_id: id(operation_id),
            required_version: ContractVersion::new(1, 0),
            inputs: if state {
                vec![
                    id("value.state"),
                    id("value.input.0"),
                    id("value.input.1"),
                    id("value.weight"),
                ]
            } else {
                vec![id("value.input.0"), id("value.input.1"), id("value.weight")]
            },
            outputs: vec![id(output)],
            attributes: BTreeMap::new(),
        };
        let (nodes, states, output) = match config.scenario.as_str() {
            "alias" => (
                vec![node(
                    "node.alias",
                    "operation.graph.alias",
                    "value.alias",
                    false,
                )],
                Vec::new(),
                id("value.alias"),
            ),
            "alias_late_consumer" => (
                vec![
                    node("node.alias", "operation.graph.alias", "value.alias", false),
                    node(
                        "node.late-consumer",
                        "operation.graph.consume",
                        "value.late",
                        false,
                    ),
                ],
                Vec::new(),
                id("value.late"),
            ),
            "state_chain" => (
                vec![
                    node(
                        "node.state-read.0",
                        "operation.graph.state-read",
                        "value.state-read.0",
                        true,
                    ),
                    node(
                        "node.state-rw.0",
                        "operation.graph.state-rw",
                        "value.state-rw.0",
                        true,
                    ),
                    node(
                        "node.state-rw.1",
                        "operation.graph.state-rw",
                        "value.state-rw.1",
                        true,
                    ),
                    node(
                        "node.state-read.1",
                        "operation.graph.state-read",
                        "value.state-read.1",
                        true,
                    ),
                ],
                vec![graph_state_spec()],
                id("value.state-read.1"),
            ),
            "state_read_only" => (
                vec![
                    node(
                        "node.state-read.0",
                        "operation.graph.state-read",
                        "value.state-read.0",
                        true,
                    ),
                    node(
                        "node.state-read.1",
                        "operation.graph.state-read",
                        "value.state-read.1",
                        true,
                    ),
                ],
                vec![graph_state_spec()],
                id("value.state-read.1"),
            ),
            _ => unreachable!(),
        };
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input.0"), id("value.input.1")],
            vec![ProgramBlock {
                id: "block.execution-graph".to_owned(),
                nodes,
            }],
            states,
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id("value.weight"),
                tensor: graph_program_tensor(),
            }],
            vec![output],
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
pub(crate) fn graph_program_tensor() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![4],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

pub(crate) fn graph_state_spec() -> StateSpec {
    StateSpec {
        id: id("state.cache"),
        value_id: id("value.state"),
        tensor: graph_program_tensor(),
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::FixedPerScope,
    }
}
