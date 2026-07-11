use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn resource_work(sequence_token_counts: &[usize]) -> ResourceWorkShape {
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

fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

fn contiguous_storage_requirement() -> DynamicStorageRequirement {
    DynamicStorageRequirement::contiguous()
}

fn storage_bindings(
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

fn contiguous_storage_bindings(
    operation: &OperationDescriptor,
) -> Vec<ProviderStorageBindingRequirement> {
    storage_bindings(operation, contiguous_storage_requirement())
}

fn paged_storage_profile(block_bytes: u64) -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena { block_bytes },
        DynamicStorageView::PagedRegions { block_bytes },
    )
    .unwrap()
}

fn paged_storage_requirement(block_bytes: u64) -> DynamicStorageRequirement {
    DynamicStorageRequirement::new(vec![paged_storage_profile(block_bytes)]).unwrap()
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

struct OrderedSchemaFamily {
    reverse: bool,
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

struct FixedSchemaFamily {
    schema: WeightSchema,
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

struct TestRegistry {
    registration: TypedFamilyRegistration<TestFamily>,
}

impl TestRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }

    fn prepare(&self) -> PreparedModelFamily {
        self.registration.prepare(&json!({"width": 4})).unwrap()
    }
}

impl ModelFamilyRegistry for TestRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

#[derive(Default)]
struct SequentialScratchFamily;

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
    catalog_with_memory(1 << 20)
}

fn catalog_with_memory(total_memory_bytes: u64) -> CapabilityCatalog {
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

fn catalog_from_operations(
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

fn catalog_with_secondary_provider() -> CapabilityCatalog {
    catalog_with_secondary_provider_storage(
        contiguous_storage_requirement(),
        contiguous_storage_requirement(),
    )
}

fn catalog_with_secondary_provider_storage(
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

struct TestOperationContract {
    descriptor: OperationDescriptor,
    calls: Arc<AtomicUsize>,
    reject_signature: bool,
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
enum EstimateBehavior {
    Correct,
    WrongEstimatorId,
    WrongEstimatorVersion,
    WrongImplementation,
    WrongInput,
    InvalidAlignment,
    MissingScratch,
}

struct TestEstimator {
    descriptor: OperationProviderDescriptor,
    calls: Arc<AtomicUsize>,
    scratch_bytes: u64,
    persistent_bytes: u64,
    behavior: EstimateBehavior,
}

struct PlanningTestRuntime;

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
        _command: Self::Command,
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
        _invocation: OperationInvocation<'_, BufferDescriptor>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

struct SequentialScratchEstimator {
    descriptor: OperationProviderDescriptor,
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
        _invocation: OperationInvocation<'_, BufferDescriptor>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

struct TestPlanningEntries {
    contracts: Vec<TestOperationContract>,
    estimators: Vec<TestEstimator>,
    contract_calls: Arc<AtomicUsize>,
    estimator_calls: Arc<AtomicUsize>,
}

impl TestPlanningEntries {
    fn new(
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

    fn build(self) -> Result<TestPlanningRegistry, VNextError> {
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

struct TestPlanningRegistry {
    registry: OperationRuntimeRegistry<PlanningTestRuntime>,
    contract_calls: Arc<AtomicUsize>,
    estimator_calls: Arc<AtomicUsize>,
}

impl TestPlanningRegistry {
    fn new(
        catalog: &CapabilityCatalog,
        scratch_bytes: u64,
        persistent_bytes: u64,
        behavior: EstimateBehavior,
    ) -> Self {
        TestPlanningEntries::new(catalog, scratch_bytes, persistent_bytes, behavior)
            .build()
            .unwrap()
    }

    fn planning(&self) -> OperationPlanningHandle<'_> {
        self.registry.planning()
    }
}

fn policy(capacity_bytes: u64) -> ResolvedRuntimePolicy {
    policy_with(capacity_bytes, 128, 3).unwrap()
}

fn policy_with(
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
struct AdversarialRuntimePolicy {
    maximum_active_sequences: u32,
    dynamic_storage_profile_order: Vec<DynamicStorageProfile>,
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

fn resolved_tensor(element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![4], element_type, ResolvedTensorLayout::Contiguous).unwrap()
}

fn binding(
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

fn resolved_values(variant: usize) -> Vec<ResolvedValueBinding> {
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

fn sequential_resolved_values(
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

fn try_node_resolution_with_registry(
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

fn node_resolution_with_registry(
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

fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &ResolvedRuntimePolicy,
    variant: usize,
    planning: &TestPlanningRegistry,
) -> PlanNodeResolution {
    node_resolution_with_registry(family, catalog, policy, variant, planning, None)
}

struct PlanFixture {
    registry: TestRegistry,
    family: PreparedModelFamily,
    catalog: CapabilityCatalog,
    policy: ResolvedRuntimePolicy,
    planning: TestPlanningRegistry,
    node_resolutions: Vec<PlanNodeResolution>,
    plan: ExecutionPlan,
}

fn plan_fixture(variant: usize) -> PlanFixture {
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

#[test]
fn operation_resource_contract_requires_explicit_presence_and_alignment() {
    assert!(ResourcePresenceRequirement::Required.accepts(true));
    assert!(!ResourcePresenceRequirement::Required.accepts(false));
    assert!(ResourcePresenceRequirement::Optional.accepts(true));
    assert!(ResourcePresenceRequirement::Optional.accepts(false));
    assert!(ResourcePresenceRequirement::Forbidden.accepts(false));
    assert!(!ResourcePresenceRequirement::Forbidden.accepts(true));

    let mut invalid_alignment = operation();
    invalid_alignment.resources.minimum_value_alignment_bytes = 3;
    assert!(invalid_alignment.validate().is_err());
    assert!(OperationProviderDescriptor::new(
        id("provider.invalid-estimator"),
        operation().id,
        operation().fingerprint().unwrap(),
        sha('f'),
        ContractVersion::new(1, 0),
        id("device.reference.0"),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation()),
        "resource estimator with spaces",
        ContractVersion::new(1, 0),
        sha('e'),
    )
    .is_err());
    assert!(EngineProviderDescriptor::new(
        id("provider.engine.invalid"),
        ContractVersion::new(1, 0),
        "not-a-sha256",
        id("device.reference.0"),
        BTreeSet::new(),
    )
    .is_err());
    assert!(OperationProviderDescriptor::new(
        id("provider.invalid-implementation"),
        operation().id,
        operation().fingerprint().unwrap(),
        "not-a-sha256",
        ContractVersion::new(1, 0),
        id("device.reference.0"),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation()),
        "resource-estimator.reference",
        ContractVersion::new(1, 0),
        sha('e'),
    )
    .is_err());
}

#[test]
fn execution_memory_is_core_owned_and_exact() {
    let fixture = plan_fixture(0);
    let memory = fixture.plan.payload().memory();
    assert_eq!(memory.device_capacity_bytes(), 1 << 20);
    assert_eq!(memory.policy_capacity_bytes(), 4096);
    assert_eq!(memory.reserve_bytes(), 128);
    assert_eq!(memory.usable_capacity_bytes(), 3968);
    assert_eq!(memory.maximum_active_sequences(), 3);
    assert_eq!(memory.static_bytes(), 48);
    assert_eq!(memory.minimum_runnable_request_bytes(), 112);
    assert_eq!(memory.theoretical_ceiling_bytes(), 384);
    assert_eq!(memory.static_allocations().len(), 2);
    assert_eq!(memory.dynamic_descriptors().len(), 4);
    assert!(!memory.dynamic_pools().is_empty());
    assert!(memory
        .dynamic_pools()
        .windows(2)
        .all(|pair| pair[0].pool_id() < pair[1].pool_id()));

    let pooled_resource_ids = memory
        .dynamic_pools()
        .iter()
        .flat_map(|pool| pool.resource_ids().iter().cloned())
        .collect::<BTreeSet<_>>();
    let dynamic_resource_ids = memory
        .dynamic_descriptors()
        .iter()
        .map(|descriptor| descriptor.base_resource_id().clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(pooled_resource_ids, dynamic_resource_ids);
    for pool in memory.dynamic_pools() {
        assert_eq!(
            pool.provisioning().mode(),
            DynamicPoolProvisioningMode::DemandDrivenElastic
        );
        assert!(
            pool.provisioning().minimum_resident_bytes()
                <= pool.provisioning().maximum_resident_bytes()
        );
        assert!(
            pool.provisioning().maximum_resident_bytes() as u128
                <= pool.theoretical_ceiling_bytes()
        );
    }
    for descriptor in memory.dynamic_descriptors() {
        let pool = memory
            .dynamic_pools()
            .iter()
            .find(|pool| pool.pool_id() == descriptor.pool_id())
            .unwrap();
        assert!(pool.resource_ids().contains(descriptor.base_resource_id()));
        assert_eq!(
            pool.compatibility().profile(),
            descriptor.storage().profile()
        );
    }

    let scratch = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
        .collect::<Vec<_>>();
    let persistent = memory
        .static_allocations()
        .iter()
        .filter(|allocation| matches!(allocation.kind(), AllocationKind::Persistent { .. }))
        .collect::<Vec<_>>();
    assert_eq!(scratch.len(), 1);
    assert!(scratch.iter().all(|descriptor| {
        descriptor.minimum_request_bytes().unwrap() == 64
            && descriptor.alignment_bytes() == 16
            && descriptor.usage() == BufferUsage::Scratch
            && descriptor.lifetime() == AllocationLifetime::Invocation
            && descriptor.theoretical_maximum_instances() == 3
    }));
    assert_eq!(persistent.len(), 1);
    assert!(persistent.iter().all(|allocation| {
        allocation.size_bytes() == 32
            && allocation.usage() == BufferUsage::Persistent
            && allocation.lifetime() == AllocationLifetime::Plan
            && allocation.storage().profile() == contiguous_storage_profile()
    }));
    let state = memory
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id().as_str() == "resource.state.0")
        .unwrap();
    assert_eq!(state.minimum_request_bytes().unwrap(), 16);
    assert_eq!(state.theoretical_maximum_instances(), 3);
    assert_eq!(state.lifetime(), AllocationLifetime::Sequence);
    assert_eq!(memory.static_buffer_requests().unwrap().len(), 2);

    let small_policy = policy(511);
    let small_plan = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &small_policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        small_plan.payload().memory().theoretical_ceiling_bytes()
            > u128::from(small_plan.payload().memory().usable_capacity_bytes())
    );
}

#[test]
fn minimum_runnable_sums_lifetime_minima_and_sequential_invocation_peak() {
    let registration = TypedFamilyRegistration::new(SequentialScratchFamily);
    let family = registration.prepare(&json!({"width": 4})).unwrap();
    let catalog = catalog();
    let policy = policy(4096);
    let descriptor = catalog.providers_for(&id("operation.main")).unwrap()[0].clone();
    let runtime_registry = OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
            calls: Arc::new(AtomicUsize::new(0)),
            reject_signature: false,
        }) as Box<dyn OperationContract>],
        vec![Box::new(SequentialScratchEstimator { descriptor })
            as Box<dyn OperationProvider<PlanningTestRuntime>>],
    )
    .unwrap();
    let planning = runtime_registry.planning();
    let first = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        id("node.first"),
        sequential_resolved_values(
            "value.input",
            "resource.sequential.input",
            "value.intermediate",
            "resource.sequential.intermediate",
        ),
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
        sequential_resolved_values(
            "value.intermediate",
            "resource.sequential.intermediate",
            "value.output",
            "resource.sequential.output",
        ),
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![first, second]).unwrap(),
    )
    .unwrap();
    let memory = plan.payload().memory();
    let scratch_sum = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
        .map(|descriptor| descriptor.minimum_request_bytes().unwrap())
        .sum::<u64>();
    assert_eq!(scratch_sum, 160);
    assert_eq!(memory.minimum_request_bytes(), 48);
    assert_eq!(memory.minimum_sequence_bytes(), 16);
    assert_eq!(memory.minimum_step_bytes(), 0);
    assert_eq!(memory.minimum_invocation_peak_bytes(), 96);
    assert_eq!(memory.minimum_runnable_request_bytes(), 160);
}

#[test]
fn runtime_capacity_reserve_and_concurrency_are_typed_planning_inputs() {
    assert!(policy_with(4096, 4096, 3).is_err());
    assert!(policy_with(4096, 128, 0).is_err());

    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let oversized = policy_with((1 << 20) + 1, 128, 3).unwrap();
    let oversized_resolution = vec![node_resolution(&family, &catalog, &oversized, 0, &planning)];
    assert!(ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &oversized, oversized_resolution).unwrap(),
    )
    .is_err());

    let four_sequences = policy_with(4096, 256, 4).unwrap();
    let resolution = vec![node_resolution(
        &family,
        &catalog,
        &four_sequences,
        0,
        &planning,
    )];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &four_sequences, resolution).unwrap(),
    )
    .unwrap();
    let memory = plan.payload().memory();
    assert_eq!(memory.maximum_active_sequences(), 4);
    assert_eq!(memory.usable_capacity_bytes(), 3840);
    assert_eq!(memory.theoretical_ceiling_bytes(), 496);
    assert!(plan.payload().nodes()[0].scratch_resource().is_some());
    let state = memory
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id().as_str() == "resource.state.0")
        .unwrap();
    assert_eq!(state.theoretical_maximum_instances(), 4);
}

#[test]
fn maximum_active_sequence_ceiling_is_nonzero_and_o_graph() {
    assert!(policy_with(1 << 20, 128, 0).is_err());
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let rejected_planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    let planning = rejected_planning.planning();
    assert!(PlanNodeResolution::resolve(
        &family,
        &catalog,
        &malicious,
        &planning,
        id("node.main"),
        resolved_values(0),
        BTreeSet::new(),
        None,
    )
    .is_err());
    assert_eq!(rejected_planning.estimator_calls.load(Ordering::SeqCst), 0);
    assert_eq!(rejected_planning.contract_calls.load(Ordering::SeqCst), 0);

    let mut expected_rows = None;
    let mut expected_provider_formula = None;
    for maximum_active_sequences in [1, 32, u32::MAX] {
        let policy = policy_with(1 << 20, 128, maximum_active_sequences).unwrap();
        let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
        let resolution = node_resolution(&family, &catalog, &policy, 0, &planning);
        let plan = ExecutionPlan::build(
            PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution.clone()]).unwrap(),
        )
        .unwrap();
        let rows = (
            plan.payload().nodes().len(),
            plan.payload().nodes()[0].resources().len(),
            plan.payload().memory().static_allocations().len(),
            plan.payload().memory().dynamic_descriptors().len(),
        );
        assert_eq!(*expected_rows.get_or_insert(rows), rows);
        let resources = &resolution.provider_resource_candidates()[0];
        let formula_evidence = (
            resources.estimator_input_fingerprint().to_owned(),
            resources.estimate_fingerprint().to_owned(),
            resources.scratch().unwrap().size_formula().clone(),
            resources.scratch().unwrap().minimum_bytes().unwrap(),
            plan.payload()
                .memory()
                .dynamic_descriptors()
                .iter()
                .find(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
                .unwrap()
                .demand()
                .clone(),
        );
        assert_eq!(
            expected_provider_formula.get_or_insert_with(|| formula_evidence.clone()),
            &formula_evidence
        );
        assert_eq!(
            plan.payload().nodes()[0]
                .scratch_resource()
                .into_iter()
                .count(),
            1
        );
        if maximum_active_sequences == u32::MAX {
            let policy_wire = serde_json::to_vec(&policy).unwrap();
            let restored: ResolvedRuntimePolicy = serde_json::from_slice(&policy_wire).unwrap();
            assert_eq!(restored.memory().maximum_active_sequences, u32::MAX);
            let restored_plan = ExecutionPlan::from_json_validated(
                &plan.to_json().unwrap(),
                &family,
                &catalog,
                &policy,
                vec![resolution],
            )
            .unwrap();
            assert_eq!(restored_plan.plan_hash(), plan.plan_hash());
        }
    }
}

#[test]
fn theoretical_ceiling_over_u64_is_canonical_evidence_not_capacity_policy() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog_with_memory(u64::MAX);
    let policy = policy_with(u64::MAX, 1, u32::MAX).unwrap();
    let planning = TestPlanningRegistry::new(
        &catalog,
        8 * 1024 * 1024 * 1024,
        32,
        EstimateBehavior::Correct,
    );
    let resolution = node_resolution(&family, &catalog, &policy, 0, &planning);
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution.clone()]).unwrap(),
    )
    .unwrap();
    assert!(plan.payload().memory().theoretical_ceiling_bytes() > u128::from(u64::MAX));
    assert!(
        u128::from(plan.payload().memory().static_bytes())
            + u128::from(plan.payload().memory().minimum_runnable_request_bytes())
            <= u128::from(plan.payload().memory().usable_capacity_bytes())
    );
    let wire = plan.to_json().unwrap();
    let value: Value = serde_json::from_slice(&wire).unwrap();
    let encoded = value["payload"]["memory"]["theoretical_ceiling_bytes"]
        .as_str()
        .unwrap();
    assert_eq!(
        encoded,
        plan.payload()
            .memory()
            .theoretical_ceiling_bytes()
            .to_string()
    );
    let restored =
        ExecutionPlan::from_json_validated(&wire, &family, &catalog, &policy, vec![resolution])
            .unwrap();
    assert_eq!(restored.plan_hash(), plan.plan_hash());
}

#[test]
fn state_capacity_demand_is_explicit_checked_and_wire_closed() {
    let state = StateSpec {
        id: id("state.scaled"),
        value_id: id("value.scaled"),
        tensor: ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::U8,
            layout: ResolvedTensorLayout::Contiguous,
        },
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::TokenScaled {
            bytes_per_token: 4,
            maximum_tokens: 128,
        },
    };
    let restored: StateSpec =
        serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
    assert_eq!(restored, state);
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "token_scaled": {"bytes_per_token": 0, "maximum_tokens": 128}
    }))
    .is_err());
    let mut undersized = serde_json::to_value(&state).unwrap();
    undersized["capacity_demand"]["token_scaled"]["bytes_per_token"] = json!(1);
    assert!(serde_json::from_value::<StateSpec>(undersized).is_err());
    assert_eq!(state.capacity_demand.theoretical_bytes(4).unwrap(), 512);
    assert!(serde_json::from_value::<StateCapacityDemand>(json!("plan_static")).is_err());
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "fixed_per_scope": {"unexpected": true}
    }))
    .is_err());
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "page_scaled": {"bytes_per_page": 4096, "maximum_pages": 1024}
    }))
    .is_err());
}

#[test]
fn provider_workspace_formulas_are_actual_shape_checked_and_wire_closed() {
    let shape = resource_work(&[3, 3, 3, 3]);
    assert_eq!(
        DynamicResourceDemand::fixed(13)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        13
    );
    assert_eq!(
        DynamicResourceDemand::actual_sequences(7, 8)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        28
    );
    assert_eq!(
        DynamicResourceDemand::tokens(3, 32)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        36
    );
    assert_eq!(
        DynamicResourceDemand::pages(11, 8)
            .unwrap()
            .evaluate_bytes(&shape)
            .is_err(),
        true
    );

    let buckets = DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(1, 8, 2, 64).unwrap(),
        DynamicResourceShapeBucket::new(4, 32, 8, 128).unwrap(),
    ])
    .unwrap();
    assert_eq!(buckets.evaluate_bytes(&shape).unwrap(), 128);
    assert!(buckets
        .evaluate_bytes(&resource_work(&[3, 3, 2, 2, 2]))
        .is_err());

    let aligned = ProviderWorkspaceRequirement::from_formula(
        DynamicResourceDemand::actual_sequences(7, 8).unwrap(),
        16,
        ProviderWorkspaceScope::Invocation,
        contiguous_storage_requirement(),
    )
    .unwrap();
    assert_eq!(aligned.evaluate_bytes(&shape).unwrap(), 32);
    assert!(ProviderWorkspaceRequirement::from_formula(
        DynamicResourceDemand::tokens(4, 32).unwrap(),
        16,
        ProviderWorkspaceScope::Plan,
        contiguous_storage_requirement(),
    )
    .is_err());
    assert!(ProviderWorkspaceRequirement::from_formula(
        DynamicResourceDemand::actual_sequences(4, 8).unwrap(),
        16,
        ProviderWorkspaceScope::Sequence,
        contiguous_storage_requirement(),
    )
    .is_err());
    assert!(ProviderWorkspaceRequirement::from_formula(
        DynamicResourceDemand::fixed(u64::MAX).unwrap(),
        16,
        ProviderWorkspaceScope::Invocation,
        contiguous_storage_requirement(),
    )
    .is_err());

    assert!(DynamicResourceDemand::tokens(u64::MAX, 2).is_err());
    assert!(DynamicResourceDemand::actual_sequences(u64::MAX, 2).is_err());
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "actual_sequences": {
                "bytes_per_sequence": u64::MAX,
                "maximum_sequences": 2
            }
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "tokens": {"bytes_per_token": 4, "maximum_tokens": 32, "unknown": 1}
        }))
        .is_err()
    );
    assert!(DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(4, 32, 8, 128).unwrap(),
        DynamicResourceShapeBucket::new(2, 64, 16, 256).unwrap(),
    ])
    .is_err());
    assert!(DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(1, 8, 2, 128).unwrap(),
        DynamicResourceShapeBucket::new(4, 32, 8, 64).unwrap(),
    ])
    .is_err());
    assert!(DynamicResourceDemand::bounded_shape_buckets(
        (1..=MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS + 1)
            .map(|index| {
                DynamicResourceShapeBucket::new(
                    index as u32,
                    index as u64,
                    index as u64,
                    index as u64,
                )
                .unwrap()
            })
            .collect(),
    )
    .is_err());

    let wire = serde_json::to_value(&aligned).unwrap();
    let restored: ProviderWorkspaceRequirement = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(restored, aligned);
    let mut unknown = wire;
    unknown["unknown"] = json!(1);
    assert!(serde_json::from_value::<ProviderWorkspaceRequirement>(unknown).is_err());
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "bounded_shape_buckets": {"buckets": [
                {"maximum_sequences": 4, "maximum_tokens": 32, "maximum_pages": 8, "bytes": 128},
                {"maximum_sequences": 2, "maximum_tokens": 64, "maximum_pages": 16, "bytes": 256}
            ]}
        }))
        .is_err()
    );
}

#[test]
fn dynamic_descriptor_and_memory_plan_standalone_wire_are_checked() {
    let fixture = plan_fixture(0);
    let mut descriptor = serde_json::to_value(
        fixture
            .plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .first()
            .unwrap(),
    )
    .unwrap();
    descriptor["alignment_bytes"] = json!(3);
    assert!(serde_json::from_value::<DynamicResourceDescriptor>(descriptor).is_err());

    let mut descriptor = serde_json::to_value(
        fixture
            .plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .first()
            .unwrap(),
    )
    .unwrap();
    descriptor["usage"] = json!("weights");
    assert!(serde_json::from_value::<DynamicResourceDescriptor>(descriptor).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    memory["minimum_invocation_peak_bytes"] = json!(u64::MAX);
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    memory["dynamic_descriptors"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    assert!(memory["dynamic_pools"].as_array().unwrap().len() > 1);
    memory["dynamic_pools"].as_array_mut().unwrap().reverse();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    let persistent = memory["static_allocations"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|allocation| allocation["usage"] == json!("persistent"))
        .unwrap();
    persistent["storage"]["profile"] = serde_json::to_value(paged_storage_profile(4096)).unwrap();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());
}

#[test]
fn execution_plan_is_deterministic_100_of_100() {
    for variant in 0..100 {
        let left = plan_fixture(variant).plan;
        let right = plan_fixture(variant).plan;
        assert_eq!(left.plan_hash(), right.plan_hash());
        assert_eq!(left.to_json().unwrap(), right.to_json().unwrap());
    }
    println!("VNEXT PLAN DETERMINISM PASS: 100/100");
}

#[test]
fn execution_plan_schema_round_trip_100_of_100() {
    for variant in 0..100 {
        let fixture = plan_fixture(variant);
        let restored = ExecutionPlan::from_json_validated(
            &fixture.plan.to_json().unwrap(),
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap();
        assert_eq!(fixture.plan, restored);
    }
    println!("VNEXT PLAN ROUNDTRIP PASS: 100/100");
}

#[test]
fn breaking_schema_versions_are_rejected_100_of_100() {
    for variant in 0..100 {
        let fixture = plan_fixture(variant);
        let mut value = serde_json::to_value(&fixture.plan).unwrap();
        value["payload"]["schema"]["major"] = json!(2);
        rehash_plan_json(&mut value);
        assert!(ExecutionPlan::from_json_validated(
            &serde_json::to_vec(&value).unwrap(),
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .is_err());
    }
    println!("VNEXT BREAKING VERSION REJECT PASS: 100/100");
}

#[test]
fn forged_self_hashed_plan_is_rejected_by_semantic_rebuild() {
    let fixture = plan_fixture(0);
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    value["payload"]["memory"]["static_allocations"][0]["size_bytes"] = json!(1024);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn externally_trusted_node_resolution_cannot_be_replaced_by_wire_data() {
    let fixture = plan_fixture(0);
    let different_resolution = vec![node_resolution(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        1,
        &fixture.planning,
    )];
    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        different_resolution,
    )
    .is_err());
}

#[test]
fn self_consistent_wire_resource_estimate_and_memory_mutation_is_rejected() {
    let fixture = plan_fixture(0);
    let alternate_planning =
        TestPlanningRegistry::new(&fixture.catalog, 96, 48, EstimateBehavior::Correct);
    let alternate_resolution = vec![node_resolution_with_registry(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        0,
        &alternate_planning,
        None,
    )];
    let alternate = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            alternate_resolution,
        )
        .unwrap(),
    )
    .unwrap();
    assert_ne!(
        alternate.payload().memory().theoretical_ceiling_bytes(),
        fixture.plan.payload().memory().theoretical_ceiling_bytes()
    );
    assert!(ExecutionPlan::from_json_validated(
        &alternate.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn self_consistent_wire_provider_selection_is_rejected() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog_with_secondary_provider();
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let original_resolution = vec![node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.reference"),
    )];
    let alternate_resolution = vec![node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.secondary"),
    )];
    let alternate = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, alternate_resolution).unwrap(),
    )
    .unwrap();
    assert_eq!(
        alternate.payload().nodes()[0]
            .selection()
            .selected_provider()
            .as_str(),
        "provider.operation.secondary"
    );
    assert!(ExecutionPlan::from_json_validated(
        &alternate.to_json().unwrap(),
        &family,
        &catalog,
        &policy,
        original_resolution,
    )
    .is_err());
}

#[test]
fn typed_planning_registry_invokes_real_contract_and_estimator_once() {
    let fixture = plan_fixture(0);
    assert_eq!(fixture.planning.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.planning.estimator_calls.load(Ordering::SeqCst), 1);
    let resources = &fixture.node_resolutions[0].provider_resource_candidates()[0];
    assert_eq!(
        resources.provider_id().as_str(),
        "provider.operation.reference"
    );
    assert_eq!(resources.estimator_id(), "resource-estimator.reference");
    let node = &fixture.plan.payload().nodes()[0];
    assert_eq!(node.provider_implementation_fingerprint(), sha('f'));
    assert_ne!(
        node.provider_implementation_fingerprint(),
        resources.estimator_implementation_fingerprint()
    );
    assert_eq!(resources.scratch().unwrap().minimum_bytes().unwrap(), 64);
    assert_eq!(resources.persistent().unwrap().minimum_bytes().unwrap(), 32);

    let _rebuilt = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.planning.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.planning.estimator_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn provider_implementation_fingerprint_is_plan_hashed_and_revalidated() {
    let fixture = plan_fixture(0);
    let original_hash = fixture.plan.plan_hash().as_str().to_owned();
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    assert_eq!(
        value["payload"]["nodes"][0]["provider_implementation_fingerprint"],
        json!(sha('f'))
    );
    value["payload"]["nodes"][0]["provider_implementation_fingerprint"] = json!(sha('0'));
    rehash_plan_json(&mut value);
    assert_ne!(value["plan_hash"], json!(original_hash));
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn planning_registry_missing_duplicate_and_mismatched_entries_fail_before_plan() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);

    let mut missing_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let missing_contract_calls = missing_contract.contract_calls.clone();
    let missing_estimator_calls = missing_contract.estimator_calls.clone();
    missing_contract.contracts.clear();
    assert!(missing_contract.build().is_err());
    assert_eq!(missing_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(missing_estimator_calls.load(Ordering::SeqCst), 0);

    let mut duplicate_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let duplicate_contract_calls = duplicate_contract.contract_calls.clone();
    let duplicate_estimator_calls = duplicate_contract.estimator_calls.clone();
    duplicate_contract.contracts.push(TestOperationContract {
        descriptor: duplicate_contract.contracts[0].descriptor.clone(),
        calls: duplicate_contract_calls.clone(),
        reject_signature: false,
    });
    assert!(duplicate_contract.build().is_err());
    assert_eq!(duplicate_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(duplicate_estimator_calls.load(Ordering::SeqCst), 0);

    let mut mismatched_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mismatched_contract_calls = mismatched_contract.contract_calls.clone();
    let mismatched_estimator_calls = mismatched_contract.estimator_calls.clone();
    mismatched_contract.contracts[0].descriptor.version = ContractVersion::new(1, 1);
    assert!(mismatched_contract.build().is_err());
    assert_eq!(mismatched_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(mismatched_estimator_calls.load(Ordering::SeqCst), 0);

    let mut rejecting_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    rejecting_contract.contracts[0].reject_signature = true;
    let rejecting_contract = rejecting_contract.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &rejecting_contract,
        None,
    )
    .is_err());
    assert_eq!(rejecting_contract.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(rejecting_contract.estimator_calls.load(Ordering::SeqCst), 0);

    let mut missing_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let missing_contract_calls = missing_estimator.contract_calls.clone();
    let missing_estimator_calls = missing_estimator.estimator_calls.clone();
    missing_estimator.estimators.clear();
    assert!(missing_estimator.build().is_err());
    assert_eq!(missing_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(missing_estimator_calls.load(Ordering::SeqCst), 0);

    let mut duplicate_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let duplicate_contract_calls = duplicate_estimator.contract_calls.clone();
    let duplicate_estimator_calls = duplicate_estimator.estimator_calls.clone();
    duplicate_estimator.estimators.push(TestEstimator {
        descriptor: duplicate_estimator.estimators[0].descriptor.clone(),
        calls: duplicate_estimator_calls.clone(),
        scratch_bytes: 64,
        persistent_bytes: 32,
        behavior: EstimateBehavior::Correct,
    });
    assert!(duplicate_estimator.build().is_err());
    assert_eq!(duplicate_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(duplicate_estimator_calls.load(Ordering::SeqCst), 0);

    let mut mismatched_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let descriptor = mismatched_estimator.estimators[0].descriptor.clone();
    mismatched_estimator.estimators[0].descriptor = OperationProviderDescriptor::new(
        descriptor.provider_id().clone(),
        descriptor.operation_id().clone(),
        descriptor.operation_fingerprint(),
        descriptor.provider_implementation_fingerprint(),
        descriptor.version(),
        descriptor.device_id().clone(),
        descriptor.capabilities().clone(),
        descriptor.accepted_weight_formats().clone(),
        descriptor.accepted_quantization_formats().clone(),
        descriptor.dynamic_storage_bindings().to_vec(),
        "resource-estimator.mismatch",
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
    )
    .unwrap();
    let mismatched_estimator = mismatched_estimator.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &mismatched_estimator,
        None,
    )
    .is_err());
    assert_eq!(
        mismatched_estimator.contract_calls.load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        mismatched_estimator.estimator_calls.load(Ordering::SeqCst),
        0
    );

    let mut mismatched_provider_implementation =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let descriptor = mismatched_provider_implementation.estimators[0]
        .descriptor
        .clone();
    mismatched_provider_implementation.estimators[0].descriptor = OperationProviderDescriptor::new(
        descriptor.provider_id().clone(),
        descriptor.operation_id().clone(),
        descriptor.operation_fingerprint(),
        sha('0'),
        descriptor.version(),
        descriptor.device_id().clone(),
        descriptor.capabilities().clone(),
        descriptor.accepted_weight_formats().clone(),
        descriptor.accepted_quantization_formats().clone(),
        descriptor.dynamic_storage_bindings().to_vec(),
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
    )
    .unwrap();
    let mismatched_provider_implementation = mismatched_provider_implementation.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &mismatched_provider_implementation,
        None,
    )
    .is_err());
    assert_eq!(
        mismatched_provider_implementation
            .contract_calls
            .load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        mismatched_provider_implementation
            .estimator_calls
            .load(Ordering::SeqCst),
        0
    );
}

#[test]
fn provider_raw_estimate_identity_input_and_output_are_revalidated_by_core() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    for behavior in [
        EstimateBehavior::WrongEstimatorId,
        EstimateBehavior::WrongEstimatorVersion,
        EstimateBehavior::WrongImplementation,
        EstimateBehavior::WrongInput,
        EstimateBehavior::InvalidAlignment,
        EstimateBehavior::MissingScratch,
    ] {
        let planning = TestPlanningRegistry::new(&catalog, 64, 32, behavior);
        assert!(
            try_node_resolution_with_registry(&family, &catalog, &policy, 0, &planning, None,)
                .is_err()
        );
        assert_eq!(planning.contract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(planning.estimator_calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn preferred_provider_is_only_a_core_validated_preference() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let resolution = try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.unregistered"),
    )
    .unwrap();
    assert_eq!(
        resolution.provider_resource_candidates()[0]
            .provider_id()
            .as_str(),
        "provider.operation.reference"
    );
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
    .unwrap();
    let selection = plan.payload().nodes()[0].selection();
    assert_eq!(
        selection.selected_provider().as_str(),
        "provider.operation.reference"
    );
    assert_eq!(
        selection.selection_reason(),
        ProviderSelectionReason::FallbackFromPreferred
    );
}

#[test]
fn storage_incompatible_preference_falls_back_with_canonical_evidence() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog_with_secondary_provider_storage(
        paged_storage_requirement(4096),
        contiguous_storage_requirement(),
    );
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let preferred: ProviderId = id("provider.operation.reference");
    let resolution = try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some(preferred.as_str()),
    )
    .unwrap();

    assert_eq!(resolution.provider_resource_candidates().len(), 1);
    assert_eq!(
        resolution.provider_resource_candidates()[0]
            .provider_id()
            .as_str(),
        "provider.operation.secondary"
    );
    let rejected_resource_ids = match resolution
        .provider_resolution_rejections()
        .get(&preferred)
        .unwrap()
    {
        PlanProviderRejectReason::StorageIncompatible { resource_ids } => resource_ids.clone(),
        other => panic!("unexpected storage rejection: {other:?}"),
    };
    assert!(!rejected_resource_ids.is_empty());
    assert!(rejected_resource_ids
        .windows(2)
        .all(|pair| pair[0] < pair[1]));

    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
    .unwrap();
    let selection = plan.payload().nodes()[0].selection();
    assert_eq!(
        selection.selected_provider().as_str(),
        "provider.operation.secondary"
    );
    assert_eq!(
        selection.selection_reason(),
        ProviderSelectionReason::FallbackFromPreferred
    );
    let rejection = selection
        .rejected_providers()
        .iter()
        .find(|rejection| rejection.provider_id() == &preferred)
        .unwrap();
    assert_eq!(
        rejection.reasons(),
        &PlanProviderRejectReason::StorageIncompatible {
            resource_ids: rejected_resource_ids,
        }
    );
}

fn exact_component(component_id: &str) -> PhysicalWeightComponentBinding {
    PhysicalWeightComponentBinding::exact_contiguous(id(component_id))
}

#[test]
fn physical_weight_layout_tree_accepts_dense_fixture() {
    let family = TestRegistry::new().prepare();
    let schema = family.weight_schema();
    schema.validate(family.family_id()).unwrap();
    let components = schema
        .physical_component_refs(&id("weight.matrix"))
        .unwrap();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].dimensions, [4]);
    assert_eq!(schema.physical_bytes(&id("weight.matrix")).unwrap(), 16);
    assert_eq!(
        schema
            .physical_resource_requirements(&id("weight.matrix"))
            .unwrap()[0]
            .physical_dimensions,
        [4]
    );
}

fn grouped_quantized_axis_index_schema() -> WeightSchema {
    let quantization = QuantizationSpec {
        format_id: id("quantization.grouped"),
        bits_per_weight: 4,
        group_size: 4,
        packing: QuantizationPacking::Linear,
        scale_type: ElementType::F16,
        zero_point_type: Some(ElementType::U8),
    };
    WeightSchema {
        format_id: id("weight-format.quantized"),
        layout_id: id("weight-layout.quantized-composite"),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.packed"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["packed.bin".to_owned()],
                dimensions: vec![4, 8],
                encoding: WeightEncoding::Quantized(quantization),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.scales"),
                role: WeightComponentRole::Scales,
                external_names: vec!["scales.bin".to_owned()],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id("component.zeros"),
                role: WeightComponentRole::ZeroPoints,
                external_names: vec!["zeros.bin".to_owned()],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id("component.axis-indices"),
                role: WeightComponentRole::Indices,
                external_names: vec!["axis-indices.bin".to_owned()],
                dimensions: vec![8],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::I32,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.quantized"),
            dimensions: vec![8, 8],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Quantized {
                packed_values: exact_component("component.packed"),
                packed_dimensions: vec![4, 8],
                scales: exact_component("component.scales"),
                zero_points: Some(exact_component("component.zeros")),
                axis_indices: Some(AxisWeightComponent {
                    component: exact_component("component.axis-indices"),
                    axis: 1,
                }),
                permutation: None,
                codebook: None,
                group_axis: 1,
                group_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    }
}

#[test]
fn physical_weight_layout_tree_accepts_grouped_quantized_axis_index_fixture() {
    let schema = grouped_quantized_axis_index_schema();
    schema.validate(&id("family.quantized")).unwrap();
    assert_eq!(
        schema
            .physical_component_refs(&id("weight.quantized"))
            .unwrap()
            .len(),
        4
    );
    assert_eq!(schema.physical_bytes(&id("weight.quantized")).unwrap(), 112);
    let resources = schema
        .physical_resource_requirements(&id("weight.quantized"))
        .unwrap()
        .into_iter()
        .map(|component| {
            (
                component.component_id,
                (component.physical_dimensions, component.resource_bytes),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(resources[&id("component.packed")], (vec![4, 8], 32));
    assert_eq!(resources[&id("component.scales")], (vec![8, 2], 32));
    assert_eq!(resources[&id("component.zeros")], (vec![8, 2], 16));
    assert_eq!(resources[&id("component.axis-indices")], (vec![8], 32));

    let mut wrong = schema.clone();
    wrong.components[1].role = WeightComponentRole::Indices;
    assert!(wrong.validate(&id("family.quantized")).is_err());

    let mut indexed = schema.clone();
    indexed.components.push(WeightComponentSpec {
        id: id("component.lookup-indices"),
        role: WeightComponentRole::Indices,
        external_names: vec!["lookup-indices.bin".to_owned()],
        dimensions: vec![8],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::U32,
        },
        required: true,
    });
    let quantized_values = indexed.tensors[0].physical_layout.clone();
    indexed.tensors[0].physical_layout = PhysicalWeightLayout::Indexed {
        indices: AxisWeightComponent {
            component: exact_component("component.lookup-indices"),
            axis: 0,
        },
        values: Box::new(quantized_values),
        source_axis_extent: 8,
    };
    indexed.validate(&id("family.indexed-quantized")).unwrap();
    assert_eq!(
        indexed
            .physical_component_refs(&id("weight.quantized"))
            .unwrap()
            .len(),
        5
    );
}

fn recursive_quantized_expert_schema() -> WeightSchema {
    let quantization = QuantizationSpec {
        format_id: id("quantization.expert-grouped"),
        bits_per_weight: 4,
        group_size: 4,
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: Some(ElementType::U8),
    };
    let mut components = Vec::new();
    let mut experts = Vec::new();
    for expert in 0..2 {
        let packed = format!("component.expert.{expert}.packed");
        let scales = format!("component.expert.{expert}.scales");
        let zero_points = format!("component.expert.{expert}.zeros");
        components.extend([
            WeightComponentSpec {
                id: id(&packed),
                role: WeightComponentRole::PackedValues,
                external_names: vec![format!("expert.{expert}.packed.bin")],
                dimensions: vec![4, 8],
                encoding: WeightEncoding::Quantized(quantization.clone()),
                required: true,
            },
            WeightComponentSpec {
                id: id(&scales),
                role: WeightComponentRole::Scales,
                external_names: vec![format!("expert.{expert}.scales.bin")],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: id(&zero_points),
                role: WeightComponentRole::ZeroPoints,
                external_names: vec![format!("expert.{expert}.zeros.bin")],
                dimensions: vec![8, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            },
        ]);
        experts.push(PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(id(packed)),
            packed_dimensions: vec![4, 8],
            scales: PhysicalWeightComponentBinding::exact_contiguous(id(scales)),
            zero_points: Some(PhysicalWeightComponentBinding::exact_contiguous(id(
                zero_points,
            ))),
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: 1,
            group_padding: PhysicalWeightPadding::Exact,
        });
    }
    WeightSchema {
        format_id: id("weight-format.expert-quantized"),
        layout_id: id("weight-layout.recursive-expert-stack"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.expert-stack"),
            dimensions: vec![2, 8, 8],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::ExpertStack {
                experts,
                expert_axis: 0,
            },
            required: true,
        }],
    }
}

#[test]
fn physical_weight_layout_tree_accepts_recursive_quantized_expert_stack_fixture() {
    let schema = recursive_quantized_expert_schema();
    schema.validate(&id("family.expert-stack")).unwrap();
    assert_eq!(
        schema
            .physical_component_refs(&id("weight.expert-stack"))
            .unwrap()
            .len(),
        6
    );
    assert_eq!(
        schema.physical_bytes(&id("weight.expert-stack")).unwrap(),
        160
    );
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id("quantization.expert-grouped")])
    );

    let encoded = serde_json::to_vec(&schema).unwrap();
    let restored: WeightSchema = serde_json::from_slice(&encoded).unwrap();
    restored.validate(&id("family.expert-stack")).unwrap();
    assert_eq!(restored, schema);

    let mut reordered_experts = schema.clone();
    let PhysicalWeightLayout::ExpertStack { experts, .. } =
        &mut reordered_experts.tensors[0].physical_layout
    else {
        unreachable!();
    };
    experts.reverse();
    reordered_experts
        .validate(&id("family.expert-stack-reordered"))
        .unwrap();
    assert_ne!(serde_json::to_vec(&reordered_experts).unwrap(), encoded);
}

#[test]
fn weight_schema_order_is_normalized_before_fingerprinting() {
    let canonical = TypedFamilyRegistration::new(OrderedSchemaFamily { reverse: false })
        .prepare(&json!({"width": 4}))
        .unwrap();
    let reversed = TypedFamilyRegistration::new(OrderedSchemaFamily { reverse: true })
        .prepare(&json!({"width": 4}))
        .unwrap();
    assert_eq!(canonical.weight_schema(), reversed.weight_schema());
    assert_eq!(
        canonical.fingerprint().unwrap(),
        reversed.fingerprint().unwrap()
    );
    assert_eq!(
        canonical.weight_schema().components[0].external_names,
        ["weight.a", "weight.z"]
    );
    let PhysicalWeightLayout::Composite { parts } = &canonical
        .weight_schema()
        .tensor(&id("weight.optional"))
        .unwrap()
        .physical_layout
    else {
        panic!("optional fixture must use a composite tree");
    };
    assert_eq!(parts[0].logical_offsets, [0]);
    assert_eq!(parts[1].logical_offsets, [2]);
}

fn blocked_schema(
    logical_shape: Vec<u64>,
    raw_storage_shape: Vec<u64>,
    tile_shape: Vec<u64>,
    axis_order: Vec<u32>,
    tile_strides_in_elements: Vec<u64>,
    padding: PhysicalWeightPadding,
) -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.blocked"),
        layout_id: id("weight-layout.blocked"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("weight.component.blocked"),
            role: WeightComponentRole::Values,
            external_names: vec!["blocked.bin".to_owned()],
            dimensions: raw_storage_shape,
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.blocked"),
            dimensions: logical_shape,
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    component_id: id("weight.component.blocked"),
                    storage: PhysicalStorageLayout::Tiled {
                        tile_shape,
                        axis_order,
                        tile_strides_in_elements,
                        padding,
                    },
                },
            },
            required: true,
        }],
    }
}

#[test]
fn blocked_weight_layout_requires_explicit_exact_or_zero_fill_padding() {
    let exact = blocked_schema(
        vec![4, 6],
        vec![6, 4],
        vec![2, 3],
        vec![1, 0],
        vec![12, 6],
        PhysicalWeightPadding::Exact,
    );
    exact.validate(&id("family.blocked")).unwrap();

    let zero_filled = blocked_schema(
        vec![5, 6],
        vec![8, 8],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![8, 8],
        },
    );
    zero_filled.validate(&id("family.blocked")).unwrap();

    let implicit_padding = blocked_schema(
        vec![5, 6],
        vec![6, 5],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::Exact,
    );
    assert!(implicit_padding.validate(&id("family.blocked")).is_err());

    let unnecessary_zero_fill = blocked_schema(
        vec![4, 8],
        vec![8, 4],
        vec![4, 4],
        vec![1, 0],
        vec![16, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![4, 8],
        },
    );
    assert!(unnecessary_zero_fill
        .validate(&id("family.blocked"))
        .is_err());

    let wrong_padded_shape = blocked_schema(
        vec![5, 6],
        vec![64],
        vec![4, 4],
        vec![1, 0],
        vec![32, 16],
        PhysicalWeightPadding::ZeroFill {
            padded_dimensions: vec![8, 12],
        },
    );
    assert!(wrong_padded_shape.validate(&id("family.blocked")).is_err());
}

#[test]
fn physical_weight_layout_tree_rejects_invalid_shape_reuse_padding_overflow_and_limits() {
    let mut wrong_axis_shape = grouped_quantized_axis_index_schema();
    wrong_axis_shape
        .components
        .iter_mut()
        .find(|component| component.id.as_str() == "component.axis-indices")
        .unwrap()
        .dimensions = vec![8, 2];
    assert!(wrong_axis_shape
        .validate(&id("family.wrong-axis-shape"))
        .is_err());

    let mut same_bytes_wrong_packed_shape = grouped_quantized_axis_index_schema();
    let PhysicalWeightLayout::Quantized {
        packed_dimensions, ..
    } = &mut same_bytes_wrong_packed_shape.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *packed_dimensions = vec![2, 16];
    assert!(same_bytes_wrong_packed_shape
        .validate(&id("family.wrong-packed-shape"))
        .is_err());

    let mut reused_component = recursive_quantized_expert_schema();
    let PhysicalWeightLayout::ExpertStack { experts, .. } =
        &mut reused_component.tensors[0].physical_layout
    else {
        unreachable!();
    };
    experts[1] = experts[0].clone();
    let error = reused_component
        .validate(&id("family.reused-component"))
        .unwrap_err();
    assert!(error.to_string().contains("referenced more than once"));

    let strided = WeightSchema {
        format_id: id("weight-format.strided"),
        layout_id: id("weight-layout.strided"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.strided"),
            role: WeightComponentRole::Values,
            external_names: vec!["strided.bin".to_owned()],
            dimensions: vec![8],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.strided"),
            dimensions: vec![2, 3],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    component_id: id("component.strided"),
                    storage: PhysicalStorageLayout::Strided {
                        strides_in_elements: vec![4, 1],
                        padding: PhysicalWeightPadding::ZeroFill {
                            padded_dimensions: vec![2, 4],
                        },
                    },
                },
            },
            required: true,
        }],
    };
    strided.validate(&id("family.strided")).unwrap();
    let mut overlapping_stride = strided.clone();
    let PhysicalWeightLayout::Stored { component } =
        &mut overlapping_stride.tensors[0].physical_layout
    else {
        unreachable!();
    };
    let PhysicalStorageLayout::Strided {
        strides_in_elements,
        ..
    } = &mut component.storage
    else {
        unreachable!();
    };
    *strides_in_elements = vec![1, 1];
    assert!(overlapping_stride
        .validate(&id("family.overlapping-stride"))
        .is_err());

    let mut overflowing = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    overflowing.components[0].dimensions = vec![u64::MAX, 2];
    overflowing.tensors[0].dimensions = vec![u64::MAX, 2];
    assert!(overflowing.validate(&id("family.overflowing")).is_err());

    let mut too_deep = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    let mut nested = too_deep.tensors[0].physical_layout.clone();
    for _ in 0..MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH {
        nested = PhysicalWeightLayout::Composite {
            parts: vec![CompositeWeightPart {
                layout: Box::new(nested),
                logical_offsets: vec![0],
                extents: vec![4],
            }],
        };
    }
    too_deep.tensors[0].physical_layout = nested;
    assert!(too_deep.validate(&id("family.too-deep")).is_err());
    assert!(too_deep
        .physical_component_refs(&id("weight.matrix"))
        .is_err());
    assert!(
        TypedFamilyRegistration::new(FixedSchemaFamily { schema: too_deep })
            .prepare(&json!({"width": 4}))
            .is_err()
    );

    let expert_count = MAX_PHYSICAL_WEIGHT_LAYOUT_NODES / 2 + 1;
    let mut components = Vec::with_capacity(expert_count);
    let mut experts = Vec::with_capacity(expert_count);
    for index in 0..expert_count {
        let component_id = format!("component.node-limit.{index:04}");
        components.push(WeightComponentSpec {
            id: id(component_id.clone()),
            role: WeightComponentRole::Values,
            external_names: vec![format!("node-limit.{index:04}.bin")],
            dimensions: vec![1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        experts.push(PhysicalWeightLayout::Dense {
            component_id: id(component_id),
        });
    }
    let too_many_nodes = WeightSchema {
        format_id: id("weight-format.node-limit"),
        layout_id: id("weight-layout.node-limit"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: id("weight.node-limit"),
            dimensions: vec![expert_count as u64, 1],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::ExpertStack {
                experts,
                expert_axis: 0,
            },
            required: true,
        }],
    };
    assert!(too_many_nodes
        .validate(&id("family.too-many-nodes"))
        .is_err());
    assert!(too_many_nodes
        .physical_component_refs(&id("weight.node-limit"))
        .is_err());
}

#[test]
fn blocked_tensor_storage_requires_explicit_exact_or_zero_fill_padding() {
    let exact = ResolvedTensorSpec::new(
        vec![4],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::Exact,
        },
    )
    .unwrap();
    assert_eq!(exact.minimum_storage_bytes().unwrap(), 16);

    assert!(ResolvedTensorSpec::new(
        vec![3],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::Exact,
        },
    )
    .is_err());

    let zero_filled = ResolvedTensorSpec::new(
        vec![3],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4],
            axis_order: vec![0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![4],
            },
        },
    )
    .unwrap();
    assert_eq!(zero_filled.minimum_storage_bytes().unwrap(), 16);

    for physical_dimensions in [vec![3], vec![8]] {
        assert!(ResolvedTensorSpec::new(
            vec![3],
            ElementType::F32,
            ResolvedTensorLayout::Blocked {
                block: vec![4],
                axis_order: vec![0],
                padding: BlockedTensorPadding::ZeroFill {
                    physical_dimensions,
                },
            },
        )
        .is_err());
    }

    let transposed = ResolvedTensorSpec::new(
        vec![3, 5],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4, 4],
            axis_order: vec![1, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![8, 4],
            },
        },
    )
    .unwrap();
    assert_eq!(transposed.minimum_storage_bytes().unwrap(), 128);
    assert!(ResolvedTensorSpec::new(
        vec![3, 5],
        ElementType::F32,
        ResolvedTensorLayout::Blocked {
            block: vec![4, 4],
            axis_order: vec![0, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![4, 8],
            },
        },
    )
    .is_err());
}

#[test]
fn model_program_rejects_duplicate_declared_outputs() {
    let family = TestRegistry::new().prepare();
    let mut value = serde_json::to_value(family.program()).unwrap();
    value["outputs"] = json!(["value.output", "value.output"]);
    assert!(serde_json::from_value::<ModelProgram>(value).is_err());
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn resolution_test_error(field: &str, reason: &str) -> VNextError {
    VNextError::InvalidResolvedModelPlan {
        field: field.to_owned(),
        reason: reason.to_owned(),
    }
}

fn decision_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSource => ResolutionDecisionSource::UserInput,
        ResolutionField::Config => ResolutionDecisionSource::ModelMetadata,
        ResolutionField::ResolvedSource
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

const RESOLUTION_CONFIG_BYTES: &[u8] = br#"{"width":4}"#;

struct LockedConfigResolutionParser;

impl ResolutionSourceParser for LockedConfigResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.test-locked-config",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('9')).unwrap(),
        )
    }

    fn parse(
        &self,
        source: ResolutionDecisionSource,
        provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        if source != ResolutionDecisionSource::ModelMetadata
            || !matches!(
                provenance,
                ResolutionSourceProvenance::LockedModelFile { relative_path }
                    if relative_path == "config.json"
            )
        {
            return Err(resolution_test_error(
                "fixture.config_parser",
                "requires model metadata from locked config.json",
            ));
        }
        let typed: Value = serde_json::from_slice(source_bytes)
            .map_err(|error| resolution_test_error("fixture.config_parser", &error.to_string()))?;
        let typed_bytes = serde_json::to_vec(&typed).unwrap();
        Ok(json!({
            "chosen": {
                "source_file": "config.json",
                "sha256": bytes_sha256(source_bytes),
                "typed_config_sha256": bytes_sha256(&typed_bytes),
            }
        }))
    }
}

static LOCKED_CONFIG_RESOLUTION_PARSER: LockedConfigResolutionParser = LockedConfigResolutionParser;

struct WrongImplementationResolutionParser;

impl ResolutionSourceParser for WrongImplementationResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.core-json",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('d')).unwrap(),
        )
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        serde_json::from_slice(source_bytes).map_err(|error| {
            resolution_test_error("fixture.wrong_implementation_parser", &error.to_string())
        })
    }
}

static WRONG_IMPLEMENTATION_RESOLUTION_PARSER: WrongImplementationResolutionParser =
    WrongImplementationResolutionParser;

struct NondeterministicResolutionParser {
    calls: AtomicUsize,
}

impl ResolutionSourceParser for NondeterministicResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.test-nondeterministic",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('7')).unwrap(),
        )
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        let mut value: Value = serde_json::from_slice(source_bytes).unwrap();
        if self.calls.fetch_add(1, Ordering::SeqCst) % 2 == 1 {
            value["chosen"] = json!("nondeterministic");
        }
        Ok(value)
    }
}

static NONDETERMINISTIC_RESOLUTION_PARSER: NondeterministicResolutionParser =
    NondeterministicResolutionParser {
        calls: AtomicUsize::new(0),
    };

fn upstream_provenance(index: usize) -> ResolutionSourceProvenance {
    ResolutionSourceProvenance::Upstream {
        producer_id: "fixture.resolution-root".to_owned(),
        producer_version: ContractVersion::new(1, 0),
        producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e')).unwrap(),
        revision: "fixture-v1".to_owned(),
        artifact_locator: format!("resolution/input/{index}"),
    }
}

fn resolved_inputs(fixture: &PlanFixture) -> ResolvedModelPlanInputs {
    let config_sha = bytes_sha256(RESOLUTION_CONFIG_BYTES);
    assert_eq!(config_sha, fixture.family.config_fingerprint());
    ResolvedModelPlanInputs {
        original_source: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/model".to_owned(),
            requested_revision: Some("main".to_owned()),
        },
        resolved_source: ResolvedModelSource {
            canonical_location: "repo/model".to_owned(),
            resolved_revision: "0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "config.json".to_owned(),
                    size_bytes: RESOLUTION_CONFIG_BYTES.len() as u64,
                    sha256: config_sha.clone(),
                },
                FileFingerprint {
                    relative_path: "template.json".to_owned(),
                    size_bytes: 30,
                    sha256: sha('c'),
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
            sha256: config_sha.clone(),
            typed_config_sha256: config_sha,
        },
        external_metadata_id: id("metadata.synthetic"),
        prepared_family: fixture.family.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.synthetic"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1000,
        },
        device: fixture.catalog.device().clone(),
        capabilities: fixture.catalog.clone(),
        runtime: fixture.policy.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.reference"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('8'),
        },
        execution_plan: fixture.plan.clone(),
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
            maximum_output_tokens: 64,
            token_ids: BTreeSet::from([3]),
            strings: vec!["stop".to_owned()],
        },
        structured_output: StructuredOutputPolicy::JsonObject,
    }
}

fn fixture_resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
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

struct ResolvedFixtureEvidence {
    inputs: ResolvedModelPlanInputs,
    bindings: Vec<ResolutionDecisionBinding>,
    source_evidence: Vec<ResolutionSourceEvidence<'static>>,
}

struct AlternateResolvedRegistry {
    registration: TypedFamilyRegistration<OrderedSchemaFamily>,
}

struct DuplicateMetadataFamily;

impl ModelFamilyProvider for DuplicateMetadataFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.duplicate-metadata"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.synthetic")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn parse_config(&self, _raw: &Value) -> Result<Self::Config, VNextError> {
        Err(VNextError::InvalidModelConfig {
            family_id: self.family_id().to_string(),
            field: "config".to_owned(),
            reason: "identity-only adversarial registration".to_owned(),
        })
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_program(&self, _config: &Self::Config) -> Result<ModelProgram, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        unreachable!("identity-only adversarial registration")
    }
}

struct CrossFamilyResolvedRegistry {
    primary: TypedFamilyRegistration<TestFamily>,
    other: TypedFamilyRegistration<GraphFamily>,
}

impl CrossFamilyResolvedRegistry {
    fn new() -> Self {
        Self {
            primary: TypedFamilyRegistration::new(TestFamily),
            other: TypedFamilyRegistration::new(GraphFamily),
        }
    }
}

impl ModelFamilyRegistry for CrossFamilyResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.primary, &self.other]
    }
}

struct DuplicateMetadataResolvedRegistry {
    primary: TypedFamilyRegistration<TestFamily>,
    duplicate: TypedFamilyRegistration<DuplicateMetadataFamily>,
}

impl DuplicateMetadataResolvedRegistry {
    fn new() -> Self {
        Self {
            primary: TypedFamilyRegistration::new(TestFamily),
            duplicate: TypedFamilyRegistration::new(DuplicateMetadataFamily),
        }
    }
}

impl ModelFamilyRegistry for DuplicateMetadataResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.primary, &self.duplicate]
    }
}

impl AlternateResolvedRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(OrderedSchemaFamily { reverse: false }),
        }
    }
}

impl ModelFamilyRegistry for AlternateResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

fn resolved_evidence(fixture: &PlanFixture) -> ResolvedFixtureEvidence {
    let inputs = resolved_inputs(fixture);
    resolved_evidence_for_inputs(inputs)
}

fn resolved_evidence_for_inputs(inputs: ResolvedModelPlanInputs) -> ResolvedFixtureEvidence {
    let mut bindings = Vec::with_capacity(RESOLUTION_FIELDS.len());
    let mut source_evidence = Vec::with_capacity(RESOLUTION_FIELDS.len());
    for (index, field) in RESOLUTION_FIELDS.into_iter().enumerate() {
        let source = decision_source(field);
        let artifact_id: ResolutionArtifactId = id(format!("artifact.{index}"));
        let field_path = "/chosen".to_owned();
        let evidence = if field == ResolutionField::Config {
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::LockedModelFile {
                    relative_path: "config.json".to_owned(),
                },
                RESOLUTION_CONFIG_BYTES.to_vec(),
                BTreeSet::from([field_path.clone()]),
                &LOCKED_CONFIG_RESOLUTION_PARSER,
            )
            .unwrap()
        } else {
            let document = json!({"chosen": fixture_resolution_value(&inputs, field)});
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                upstream_provenance(index),
                serde_json::to_vec(&document).unwrap(),
                BTreeSet::from([field_path.clone()]),
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap()
        };
        source_evidence.push(evidence);
        bindings.push(
            ResolutionDecisionBinding::new(
                field,
                source,
                id(format!("reason.{index}")),
                artifact_id,
                field_path,
            )
            .unwrap(),
        );
    }
    ResolvedFixtureEvidence {
        inputs,
        bindings,
        source_evidence,
    }
}

#[test]
fn resolved_model_plan_closes_all_contract_links() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &context).unwrap();
    let restored =
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &context).unwrap();
    assert_eq!(plan, restored);
    assert_eq!(plan.execution_plan(), &fixture.plan);
    assert_eq!(plan.parts().engine.implementation_fingerprint, sha('8'));
    assert_eq!(plan.parts().decisions().len(), RESOLUTION_FIELDS.len());
    let locked = plan
        .parts()
        .source_artifacts()
        .iter()
        .find(|artifact| artifact.source() == ResolutionDecisionSource::ModelMetadata)
        .unwrap();
    assert!(matches!(
        locked.provenance(),
        ResolutionSourceProvenance::LockedModelFile { relative_path }
            if relative_path == "config.json"
    ));
    assert_eq!(
        locked.content_size_bytes(),
        RESOLUTION_CONFIG_BYTES.len() as u64
    );

    for mutate in [
        |value: &mut Value| {
            value["parts"]["prepared_family"]["canonical_config"]["width"] = json!(8)
        },
        |value: &mut Value| {
            value["parts"]["engine"]["provider_id"] = json!("provider.engine.unknown")
        },
        |value: &mut Value| {
            value["parts"]["engine"]["implementation_fingerprint"] = json!(sha('0'))
        },
        |value: &mut Value| value["parts"]["stop"]["token_ids"] = json!([1000]),
        |value: &mut Value| value["parts"]["device"]["unknown_nested_field"] = json!(true),
        |value: &mut Value| {
            value["parts"]["capabilities"]["operations"]["operation.main"]["unknown_nested_field"] =
                json!(true)
        },
        |value: &mut Value| {
            value["parts"]["execution_plan"]["payload"]["memory"]["unknown_nested_field"] =
                json!(true)
        },
    ] {
        let mut tampered = serde_json::to_value(&plan).unwrap();
        mutate(&mut tampered);
        assert!(ResolvedModelPlan::from_json_validated(
            &serde_json::to_vec(&tampered).unwrap(),
            &context,
        )
        .is_err());
    }
}

#[test]
fn resolved_model_plan_initial_construction_requires_verified_evidence_context() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);

    let missing_sources = &evidence.source_evidence[1..];
    let missing_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        missing_sources,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &missing_context,
    )
    .is_err());

    let mut extra_sources = evidence.source_evidence.clone();
    extra_sources.push(
        ResolutionSourceEvidence::new(
            id("artifact.unused"),
            ResolutionDecisionSource::ProductDefault,
            upstream_provenance(999),
            serde_json::to_vec(&json!({"chosen": "unused"})).unwrap(),
            BTreeSet::from(["/chosen".to_owned()]),
            &JSON_RESOLUTION_SOURCE_PARSER,
        )
        .unwrap(),
    );
    let extra_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &extra_sources,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &extra_context).is_err());
}

#[test]
fn resolved_source_evidence_rejects_raw_bytes_and_provenance_tampering() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();

    let first = &evidence.source_evidence[0];
    let first_document: Value = serde_json::from_slice(first.source_bytes()).unwrap();
    let mut wrong_bytes = evidence.source_evidence.clone();
    wrong_bytes[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        first.provenance().clone(),
        serde_json::to_vec_pretty(&first_document).unwrap(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let wrong_bytes_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_bytes,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &wrong_bytes_context,)
            .is_err()
    );

    let mut wrong_provenance = evidence.source_evidence.clone();
    let mut changed_provenance = upstream_provenance(0);
    let ResolutionSourceProvenance::Upstream { revision, .. } = &mut changed_provenance else {
        unreachable!();
    };
    *revision = "fixture-v2".to_owned();
    wrong_provenance[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        changed_provenance,
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let wrong_provenance_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_provenance,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::from_json_validated(
        &plan.to_json().unwrap(),
        &wrong_provenance_context,
    )
    .is_err());
    let mut invalid_upstream = upstream_provenance(0);
    let ResolutionSourceProvenance::Upstream { revision, .. } = &mut invalid_upstream else {
        unreachable!();
    };
    *revision = "not canonical".to_owned();
    assert!(ResolutionSourceEvidence::new(
        id("artifact.invalid-upstream"),
        first.source(),
        invalid_upstream,
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .is_err());

    let config_index = RESOLUTION_FIELDS
        .iter()
        .position(|field| *field == ResolutionField::Config)
        .unwrap();
    let config_document = json!({
        "chosen": fixture_resolution_value(&evidence.inputs, ResolutionField::Config)
    });
    let mut unlocked_model_metadata = evidence.source_evidence.clone();
    unlocked_model_metadata[config_index] = ResolutionSourceEvidence::new(
        unlocked_model_metadata[config_index].id().clone(),
        ResolutionDecisionSource::ModelMetadata,
        upstream_provenance(config_index),
        serde_json::to_vec(&config_document).unwrap(),
        BTreeSet::from(["/chosen".to_owned()]),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let unlocked_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &unlocked_model_metadata,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &unlocked_context,
    )
    .is_err());

    let mut missing_locked_file = evidence.source_evidence.clone();
    missing_locked_file[config_index] = ResolutionSourceEvidence::new(
        missing_locked_file[config_index].id().clone(),
        ResolutionDecisionSource::ModelMetadata,
        ResolutionSourceProvenance::LockedModelFile {
            relative_path: "missing.json".to_owned(),
        },
        serde_json::to_vec(&config_document).unwrap(),
        BTreeSet::from(["/chosen".to_owned()]),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let missing_locked_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &missing_locked_file,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(
        ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &missing_locked_context,)
            .is_err()
    );

    let mut tampered_size = serde_json::to_value(&plan).unwrap();
    let locked_artifact = tampered_size["parts"]["source_artifacts"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|artifact| artifact["provenance"]["kind"] == json!("locked_model_file"))
        .unwrap();
    locked_artifact["content_size_bytes"] = json!(RESOLUTION_CONFIG_BYTES.len() as u64 + 1);
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&tampered_size).unwrap(),
        &context,
    )
    .is_err());
}

#[test]
fn resolved_source_parser_identity_and_determinism_are_enforced() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let first = &evidence.source_evidence[0];
    assert!(ResolutionSourceEvidence::new(
        id("artifact.nondeterministic"),
        first.source(),
        first.provenance().clone(),
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &NONDETERMINISTIC_RESOLUTION_PARSER,
    )
    .unwrap()
    .validate()
    .is_err());

    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();
    let mut wrong_implementation = evidence.source_evidence.clone();
    wrong_implementation[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        first.provenance().clone(),
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &WRONG_IMPLEMENTATION_RESOLUTION_PARSER,
    )
    .unwrap();
    let wrong_implementation_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_implementation,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::from_json_validated(
        &plan.to_json().unwrap(),
        &wrong_implementation_context,
    )
    .is_err());

    let mut wrong_parser_id = serde_json::to_value(&plan).unwrap();
    wrong_parser_id["parts"]["source_artifacts"][0]["parser"]["id"] =
        json!("resolution-parser.other");
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_id).unwrap(),
        &context,
    )
    .is_err());
    let mut wrong_parser_version = serde_json::to_value(&plan).unwrap();
    wrong_parser_version["parts"]["source_artifacts"][0]["parser"]["version"]["minor"] = json!(1);
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_version).unwrap(),
        &context,
    )
    .is_err());
    let mut wrong_parser_implementation = serde_json::to_value(&plan).unwrap();
    wrong_parser_implementation["parts"]["source_artifacts"][0]["parser"]
        ["implementation_fingerprint"] = json!(sha('0'));
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_implementation).unwrap(),
        &context,
    )
    .is_err());
}

#[test]
fn resolved_external_device_catalog_runtime_and_node_resolution_are_exact() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();

    let wrong_node_resolutions = vec![node_resolution(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        99,
        &fixture.planning,
    )];
    let wrong_node_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &wrong_node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_node_context,
    )
    .is_err());
    assert!(
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &wrong_node_context,)
            .is_err()
    );

    let alternate_registry = AlternateResolvedRegistry::new();
    let wrong_registry_context = ResolvedPlanValidationContext::new(
        &alternate_registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_registry_context,
    )
    .is_err());

    let mut wrong_device = fixture.catalog.device().clone();
    wrong_device.total_memory_bytes += 1;
    let wrong_device_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        &wrong_device,
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_device_context,
    )
    .is_err());

    let wrong_catalog = catalog_with_secondary_provider();
    let wrong_catalog_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        wrong_catalog.device(),
        &wrong_catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_catalog_context,
    )
    .is_err());

    let wrong_runtime = policy(4095);
    let wrong_runtime_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &wrong_runtime,
    );
    assert!(
        ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &wrong_runtime_context,)
            .is_err()
    );
}

#[test]
fn resolved_model_family_identity_is_unique_and_fail_closed() {
    const EXPECTED: usize = 5;
    let fixture = plan_fixture(0);
    let mut passed = 0usize;

    let mut unknown_inputs = resolved_inputs(&fixture);
    unknown_inputs.external_metadata_id = id("metadata.unknown");
    let unknown = resolved_evidence_for_inputs(unknown_inputs);
    let unknown_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &unknown.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(unknown.inputs, unknown.bindings, &unknown_context),
        Err(VNextError::UnknownExternalModelMetadata { .. })
    ));
    passed += 1;

    let cross_registry = CrossFamilyResolvedRegistry::new();
    let mut wrong_family_inputs = resolved_inputs(&fixture);
    wrong_family_inputs.external_metadata_id = id("metadata.execution-graph");
    let wrong_family = resolved_evidence_for_inputs(wrong_family_inputs);
    let wrong_family_context = ResolvedPlanValidationContext::new(
        &cross_registry,
        &wrong_family.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(
            wrong_family.inputs,
            wrong_family.bindings,
            &wrong_family_context,
        ),
        Err(VNextError::InvalidResolvedModelPlan {
            field,
            ..
        }) if field == "external_metadata_id"
    ));
    passed += 1;

    let mut alias_inputs = resolved_inputs(&fixture);
    alias_inputs.external_metadata_id = id("metadata.synthetic.alias");
    let alias = resolved_evidence_for_inputs(alias_inputs);
    let alias_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &alias.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(alias.inputs, alias.bindings, &alias_context),
        Err(VNextError::InvalidResolvedModelPlan {
            field,
            ..
        }) if field == "external_metadata_id"
    ));
    passed += 1;

    let duplicate_registry = DuplicateMetadataResolvedRegistry::new();
    let duplicate = resolved_evidence(&fixture);
    let duplicate_context = ResolvedPlanValidationContext::new(
        &duplicate_registry,
        &duplicate.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(
            duplicate.inputs.clone(),
            duplicate.bindings.clone(),
            &duplicate_context,
        ),
        Err(VNextError::AmbiguousModelFamilyRegistration {
            identity_kind: "external metadata",
            matches: 2,
            ..
        })
    ));
    passed += 1;

    let trusted_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &duplicate.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let trusted =
        ResolvedModelPlan::new(duplicate.inputs, duplicate.bindings, &trusted_context).unwrap();
    assert!(ResolvedModelPlan::from_json_validated(
        &trusted.to_json().unwrap(),
        &duplicate_context,
    )
    .is_err());
    passed += 1;

    assert_eq!(passed, EXPECTED);
    println!("VNEXT MODEL IDENTITY PASS: {passed}/{EXPECTED}");
}

#[test]
fn resolution_source_matrix_rejects_forbidden_binding_before_plan() {
    assert!(ResolutionDecisionBinding::new(
        ResolutionField::Family,
        ResolutionDecisionSource::ProductDefault,
        id("reason.forbidden"),
        id("artifact.forbidden"),
        "/chosen",
    )
    .is_err());
}

fn rejected_plan_mutation(fixture: &PlanFixture, mutate: impl FnOnce(&mut Value)) {
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    mutate(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn unknown_inputs_fail_closed() {
    const EXPECTED: usize = 62;
    let fixture = plan_fixture(0);
    let mut passed = 0;
    macro_rules! reject {
        ($body:expr) => {{
            rejected_plan_mutation(&fixture, $body);
            passed += 1;
        }};
    }

    reject!(|value| value["payload"]["schema"]["major"] = json!(9));
    reject!(|value| value["payload"]["family_id"] = json!("family.other"));
    reject!(|value| value["payload"]["device_id"] = json!("device.other"));
    reject!(|value| value["payload"]["prepared_family_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["program_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["capability_catalog_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["policy_version"]["major"] = json!(0));
    reject!(|value| value["payload"]["policy_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["weight_format"] = json!("weight-format.other"));
    reject!(|value| value["payload"]["quantization_formats"] = json!(["quantization.other"]));
    reject!(|value| value["payload"]["nodes"] = json!([]));
    reject!(|value| value["payload"]["nodes"][0]["operation_id"] = json!("operation.unknown"));
    reject!(|value| value["payload"]["nodes"][0]["operation_fingerprint"] = json!(sha('0')));
    reject!(
        |value| value["payload"]["nodes"][0]["required_capabilities"] =
            json!(["capability.missing"])
    );
    reject!(
        |value| value["payload"]["nodes"][0]["selection"]["selected_provider"] =
            json!("provider.unknown")
    );
    reject!(
        |value| value["payload"]["nodes"][0]["selection"]["selection_reason"] =
            json!("fallback_from_preferred")
    );
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["provider_id"] =
            json!("provider.operation.unknown")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_id"] =
            json!("resource-estimator.other")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_version"]["major"] = json!(2)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_implementation_fingerprint"] =
            json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_input_fingerprint"] =
            json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimate_fingerprint"] = json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["value_alignment_bytes"] = json!(8)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["size_formula"]["fixed"]
            ["bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["alignment_bytes"] = json!(8)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["scope"] = json!("plan")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["persistent"]["size_formula"]["fixed"]
            ["bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["persistent"]["scope"] =
            json!("sequence")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["values"]
            .as_array_mut()
            .unwrap()
            .pop();
    });
    reject!(|value| value["payload"]["nodes"][0]["values"][0]["tensor"]["dimensions"] = json!([1]));
    reject!(
        |value| value["payload"]["nodes"][0]["values"][0]["storage"]["components"][0]
            ["length_bytes"] = json!(1)
    );
    reject!(|value| value["payload"]["nodes"][0]["scratch_resource"] = Value::Null);
    reject!(|value| value["payload"]["nodes"][0]["persistent_resource"] = Value::Null);
    reject!(|value| value["payload"]["nodes"][0]["resources"] = json!([]));
    reject!(|value| value["payload"]["memory"]["device_capacity_bytes"] = json!(1));
    reject!(|value| value["payload"]["memory"]["policy_capacity_bytes"] = json!(1));
    reject!(|value| value["payload"]["memory"]["reserve_bytes"] = json!(129));
    reject!(|value| value["payload"]["memory"]["usable_capacity_bytes"] = json!(3967));
    reject!(|value| value["payload"]["memory"]["maximum_active_sequences"] = json!(2));
    reject!(|value| value["payload"]["memory"]["theoretical_ceiling_bytes"] = json!("1"));
    reject!(|value| value["payload"]["memory"]["static_allocations"] = json!([]));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["per_instance_bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["instance_stride_bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["instance_count"] = json!(2)
    });
    reject!(|value| value["payload"]["memory"]["static_allocations"][0]["size_bytes"] = json!(1));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["alignment_bytes"] = json!(8)
    });
    reject!(
        |value| value["payload"]["memory"]["static_allocations"][0]["usage"] = json!("transfer")
    );
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["element_type"] = json!("u32")
    });
    reject!(|value| value["payload"]["plan_id"] = json!("plan/forged"));
    reject!(|value| value["plan_hash"] = json!(sha('0')));
    reject!(|value| value["unknown_field"] = json!(true));
    reject!(|value| value["payload"]["memory"]["unknown_nested_field"] = json!(true));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["unknown_nested_field"] = json!(true)
    });

    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &policy(4095),
        fixture.node_resolutions.clone(),
    )
    .is_err());
    passed += 1;
    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        vec![node_resolution(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            7,
            &fixture.planning,
        )],
    )
    .is_err());
    passed += 1;
    assert!((&fixture.registry as &dyn ModelFamilyRegistry)
        .resolve(&id("family.unknown"))
        .is_err());
    passed += 1;

    assert!(policy_with(1 << 20, 128, 0).is_err());
    passed += 1;

    let mut zero_policy_wire = serde_json::to_value(&fixture.policy).unwrap();
    zero_policy_wire["memory"]["maximum_active_sequences"] = json!(0);
    assert!(serde_json::from_value::<ResolvedRuntimePolicy>(zero_policy_wire).is_err());
    passed += 1;

    let rejected_planning =
        TestPlanningRegistry::new(&fixture.catalog, 64, 32, EstimateBehavior::Correct);
    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    let planning = rejected_planning.planning();
    assert!(PlanNodeResolution::resolve(
        &fixture.family,
        &fixture.catalog,
        &malicious,
        &planning,
        id("node.main"),
        resolved_values(0),
        BTreeSet::new(),
        None,
    )
    .is_err());
    assert_eq!(rejected_planning.estimator_calls.load(Ordering::SeqCst), 0);
    passed += 1;

    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    assert!(ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &malicious,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .is_err());
    passed += 1;

    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "token_scaled": {"bytes_per_token": 0, "maximum_tokens": 1}
    }))
    .is_err());
    passed += 1;

    let invalid_state = json!({
        "id": "state.invalid-demand",
        "value_id": "value.invalid-demand",
        "tensor": {
            "dimensions": [4],
            "element_type": "u8",
            "layout": "contiguous"
        },
        "lifetime": "sequence",
        "capacity_demand": {
            "page_scaled": {"bytes_per_page": 1, "maximum_pages": 1}
        }
    });
    assert!(serde_json::from_value::<StateSpec>(invalid_state).is_err());
    passed += 1;

    assert_eq!(passed, EXPECTED);
    println!("VNEXT FAIL CLOSED PASS: {passed}/{EXPECTED}");
}

#[test]
fn provider_catalog_and_reference_oracle_fail_closed() {
    let base = operation();
    let mut missing = base.clone();
    missing.oracle = OracleSpec::ReferenceOperation {
        operation_id: id("operation.missing"),
        version: ContractVersion::new(1, 0),
    };
    let device = catalog().device().clone();
    assert!(CapabilityCatalog::new(
        device,
        vec![missing.clone()],
        BTreeMap::from([(missing.id.clone(), Vec::new())]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.reference"),
            ContractVersion::new(1, 0),
            sha('8'),
            id("device.reference.0"),
            BTreeSet::from([id("capability.compute")]),
        )
        .unwrap()],
    )
    .is_err());

    let report = catalog()
        .provider_compatibility(
            ProviderCompatibilityRequest::new(
                base.id.clone(),
                ContractVersion::new(1, 0),
                BTreeSet::new(),
                BTreeSet::from([id("weight-format.unsupported")]),
                BTreeSet::new(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(report.compatible_provider_ids().is_empty());
    assert_eq!(report.rejected().len(), 1);
    assert!(matches!(
        report.rejected()[0].reasons.as_slice(),
        [ProviderCompatibilityRejectReason::UnsupportedWeightFormats { .. }]
    ));

    let mut reference = base.clone();
    reference.id = id("operation.oracle.reference");
    reference.oracle = OracleSpec::Exact;
    let mut incompatible = base.clone();
    incompatible.id = id("operation.oracle.incompatible");
    incompatible.outputs = vec![tensor_contract(
        ElementType::U8,
        TensorAccess::Write,
        AliasPolicy::NoAlias,
    )];
    incompatible.oracle = OracleSpec::ReferenceOperation {
        operation_id: reference.id.clone(),
        version: ContractVersion::new(1, 0),
    };
    assert!(catalog_from_operations(vec![incompatible, reference]).is_err());

    let mut cycle_a = base.clone();
    cycle_a.id = id("operation.oracle.cycle-a");
    cycle_a.oracle = OracleSpec::ReferenceOperation {
        operation_id: id("operation.oracle.cycle-b"),
        version: ContractVersion::new(1, 0),
    };
    let mut cycle_b = base.clone();
    cycle_b.id = id("operation.oracle.cycle-b");
    cycle_b.oracle = OracleSpec::ReferenceOperation {
        operation_id: cycle_a.id.clone(),
        version: ContractVersion::new(1, 0),
    };
    assert!(catalog_from_operations(vec![cycle_a, cycle_b]).is_err());

    let oracle_chain = |length: usize| {
        (0..length)
            .map(|index| {
                let mut descriptor = base.clone();
                descriptor.id = id(format!("operation.oracle.chain.{index:04}"));
                descriptor.oracle = if index + 1 == length {
                    OracleSpec::Exact
                } else {
                    OracleSpec::ReferenceOperation {
                        operation_id: id(format!("operation.oracle.chain.{:04}", index + 1)),
                        version: ContractVersion::new(1, 0),
                    }
                };
                descriptor
            })
            .collect::<Vec<_>>()
    };
    catalog_from_operations(oracle_chain(MAX_REFERENCE_ORACLE_DEPTH)).unwrap();
    assert!(catalog_from_operations(oracle_chain(MAX_REFERENCE_ORACLE_DEPTH + 1)).is_err());

    let too_many_operations = (0..=MAX_OPERATION_CATALOG_ROWS)
        .map(|index| {
            let mut descriptor = base.clone();
            descriptor.id = id(format!("operation.oracle.row-limit.{index:05}"));
            descriptor
        })
        .collect::<Vec<_>>();
    assert!(CapabilityCatalog::new(
        catalog().device().clone(),
        too_many_operations,
        BTreeMap::from([(id("operation.placeholder"), Vec::new())]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.row-limit"),
            ContractVersion::new(1, 0),
            sha('8'),
            id("device.reference.0"),
            BTreeSet::from([id("capability.compute")]),
        )
        .unwrap()],
    )
    .is_err());
}

#[test]
fn prepared_family_wire_requires_typed_registry_reconstruction() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let bytes = serde_json::to_vec(&family).unwrap();
    let unvalidated = PreparedModelFamily::decode_untrusted(&bytes).unwrap();
    assert_eq!(unvalidated.revalidate(&registry).unwrap(), family);

    let mut value = serde_json::to_value(&family).unwrap();
    value["program"]["outputs"] = json!(["value.input"]);
    let unvalidated =
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(unvalidated.revalidate(&registry).is_err());
}

#[test]
fn mandatory_object_safe_contracts_accept_trait_objects() {
    fn boundaries(
        _runtime: &dyn DeviceRuntime<
            Buffer = Vec<u8>,
            Stream = (),
            Command = (),
            Fence = (),
            Error = std::io::Error,
        >,
        _operation: &dyn OperationContract,
        _estimator: &dyn OperationResourceEstimator,
        _planning: &dyn OperationPlanningRegistry,
        _registration: &dyn ModelFamilyRegistration,
        _registry: &dyn ModelFamilyRegistry,
        _events: &dyn ExecutionEventSink,
    ) {
    }
    let _ = boundaries;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct GraphConfig {
    scenario: String,
}

struct GraphFamily;

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

fn graph_program_tensor() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![4],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

fn graph_state_spec() -> StateSpec {
    StateSpec {
        id: id("state.cache"),
        value_id: id("value.state"),
        tensor: graph_program_tensor(),
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::FixedPerScope,
    }
}

fn graph_operation(
    operation_id: &str,
    state_access: Option<TensorAccess>,
    output_alias: AliasPolicy,
) -> OperationDescriptor {
    let mut inputs = Vec::new();
    if let Some(access) = state_access {
        inputs.push(tensor_contract(
            ElementType::F32,
            access,
            AliasPolicy::NoAlias,
        ));
    }
    inputs.extend([
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
    ]);
    OperationDescriptor {
        id: id(operation_id),
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![tensor_contract(
            ElementType::F32,
            TensorAccess::Write,
            output_alias,
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

fn graph_catalog(alias_policy: AliasPolicy) -> CapabilityCatalog {
    let operations = vec![
        graph_operation("operation.graph.alias", None, alias_policy),
        graph_operation("operation.graph.consume", None, AliasPolicy::NoAlias),
        graph_operation(
            "operation.graph.state-read",
            Some(TensorAccess::Read),
            AliasPolicy::NoAlias,
        ),
        graph_operation(
            "operation.graph.state-rw",
            Some(TensorAccess::ReadWrite),
            AliasPolicy::NoAlias,
        ),
    ];
    let device_id: DeviceId = id("device.execution-graph.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let providers = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| {
            (
                operation.id.clone(),
                vec![OperationProviderDescriptor::new(
                    id(format!("provider.operation.graph.{index}")),
                    operation.id.clone(),
                    operation.fingerprint().unwrap(),
                    sha(char::from(b'1' + index as u8)),
                    ContractVersion::new(1, 0),
                    device_id.clone(),
                    capabilities.clone(),
                    BTreeSet::from([id("weight-format.execution-graph")]),
                    BTreeSet::new(),
                    contiguous_storage_bindings(operation),
                    format!("resource-estimator.graph.{index}"),
                    ContractVersion::new(1, 0),
                    sha(char::from(b'5' + index as u8)),
                )
                .unwrap()],
            )
        })
        .collect::<BTreeMap<_, _>>();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: sha('a'),
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        operations,
        providers,
        vec![EngineProviderDescriptor::new(
            id("provider.engine.execution-graph"),
            ContractVersion::new(1, 0),
            sha('b'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
enum GraphAliasStorage {
    Distinct,
    ExactTarget,
    PartialTarget,
    ExactWrongInput,
}

fn graph_value_binding(
    value_id: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    resource_id: &str,
    offset_bytes: u64,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id(value_id),
        role,
        ordinal,
        resolved_tensor(ElementType::F32),
        access,
        alias,
        usage,
        ResolvedValueStorage::single(id(resource_id), offset_bytes, 16, ElementType::F32).unwrap(),
    )
    .unwrap()
}

fn graph_weight_binding(ordinal: u32) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id("value.weight"),
        ResolvedValueRole::Input,
        ordinal,
        resolved_tensor(ElementType::F32),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
            Some(id("weight.component")),
            id("resource.graph.weight"),
            0,
            16,
            ElementType::F32,
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

fn graph_node_bindings(
    node: &ProgramNode,
    alias_policy: &AliasPolicy,
    alias_storage: GraphAliasStorage,
) -> Vec<ResolvedValueBinding> {
    match node.operation_id.as_str() {
        "operation.graph.alias" => {
            let (resource, offset) = match alias_storage {
                GraphAliasStorage::Distinct => ("resource.graph.alias", 0),
                GraphAliasStorage::ExactTarget => ("resource.graph.input.0", 0),
                GraphAliasStorage::PartialTarget => ("resource.graph.input.0", 8),
                GraphAliasStorage::ExactWrongInput => ("resource.graph.input.1", 0),
            };
            vec![
                graph_value_binding(
                    "value.input.0",
                    ResolvedValueRole::Input,
                    0,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.0",
                    0,
                ),
                graph_value_binding(
                    "value.input.1",
                    ResolvedValueRole::Input,
                    1,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.1",
                    0,
                ),
                graph_weight_binding(2),
                graph_value_binding(
                    "value.alias",
                    ResolvedValueRole::Output,
                    0,
                    TensorAccess::Write,
                    alias_policy.clone(),
                    BufferUsage::Activations,
                    resource,
                    offset,
                ),
            ]
        }
        "operation.graph.consume" => vec![
            graph_value_binding(
                "value.input.0",
                ResolvedValueRole::Input,
                0,
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.input.0",
                0,
            ),
            graph_value_binding(
                "value.input.1",
                ResolvedValueRole::Input,
                1,
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.input.1",
                0,
            ),
            graph_weight_binding(2),
            graph_value_binding(
                "value.late",
                ResolvedValueRole::Output,
                0,
                TensorAccess::Write,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.late",
                0,
            ),
        ],
        "operation.graph.state-read" | "operation.graph.state-rw" => {
            let state_access = if node.operation_id.as_str() == "operation.graph.state-read" {
                TensorAccess::Read
            } else {
                TensorAccess::ReadWrite
            };
            vec![
                graph_value_binding(
                    "value.state",
                    ResolvedValueRole::Input,
                    0,
                    state_access,
                    AliasPolicy::NoAlias,
                    BufferUsage::State,
                    "resource.graph.state",
                    0,
                ),
                graph_value_binding(
                    "value.input.0",
                    ResolvedValueRole::Input,
                    1,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.0",
                    0,
                ),
                graph_value_binding(
                    "value.input.1",
                    ResolvedValueRole::Input,
                    2,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.1",
                    0,
                ),
                graph_weight_binding(3),
                graph_value_binding(
                    node.outputs[0].as_str(),
                    ResolvedValueRole::Output,
                    0,
                    TensorAccess::Write,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    &format!("resource.graph.{}", node.outputs[0].as_str()),
                    0,
                ),
            ]
        }
        _ => unreachable!(),
    }
}

struct GraphPlanFixture {
    family: PreparedModelFamily,
    catalog: CapabilityCatalog,
    policy: ResolvedRuntimePolicy,
    resolutions: Vec<PlanNodeResolution>,
    plan: ExecutionPlan,
}

fn graph_plan_fixture(
    scenario: &str,
    alias_policy: AliasPolicy,
    alias_storage: GraphAliasStorage,
) -> Result<GraphPlanFixture, VNextError> {
    let family =
        TypedFamilyRegistration::new(GraphFamily).prepare(&json!({"scenario": scenario}))?;
    let catalog = graph_catalog(alias_policy.clone());
    let policy = policy(16 * 1024);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut resolutions = Vec::new();
    for node in family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
    {
        let planning_handle = planning.planning();
        resolutions.push(PlanNodeResolution::resolve(
            &family,
            &catalog,
            &policy,
            &planning_handle,
            node.id.clone(),
            graph_node_bindings(node, &alias_policy, alias_storage),
            BTreeSet::new(),
            None,
        )?);
    }
    let plan = ExecutionPlan::build(PlanBuildRequest::new(
        &family,
        &catalog,
        &policy,
        resolutions.clone(),
    )?)?;
    Ok(GraphPlanFixture {
        family,
        catalog,
        policy,
        resolutions,
        plan,
    })
}

#[test]
fn execution_alias_must_alias_builds_exact_equivalence_and_single_allocation() {
    let fixture = graph_plan_fixture(
        "alias",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    let node = &fixture.plan.payload().nodes()[0];
    assert_eq!(node.exact_aliases().len(), 1);
    assert_eq!(
        node.exact_aliases()[0].kind(),
        PlanExactAliasKind::MustAlias
    );
    assert_eq!(
        node.exact_aliases()[0].input_value_id().as_str(),
        "value.input.0"
    );
    assert_eq!(
        node.exact_aliases()[0].output_value_id().as_str(),
        "value.alias"
    );
    let memory = fixture.plan.payload().memory();
    let matching_static = memory
        .static_allocations()
        .iter()
        .filter(|allocation| allocation.resource_id().as_str() == "resource.graph.input.0")
        .count();
    let matching_dynamic = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.base_resource_id().as_str() == "resource.graph.input.0")
        .count();
    assert_eq!(matching_static + matching_dynamic, 1);
}

#[test]
fn execution_alias_may_alias_supports_distinct_or_exact_storage() {
    let distinct = graph_plan_fixture(
        "alias",
        AliasPolicy::MayAlias { tensor_index: 0 },
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    assert!(distinct.plan.payload().nodes()[0]
        .exact_aliases()
        .is_empty());

    let exact = graph_plan_fixture(
        "alias",
        AliasPolicy::MayAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    assert_eq!(exact.plan.payload().nodes()[0].exact_aliases().len(), 1);
    assert_eq!(
        exact.plan.payload().nodes()[0].exact_aliases()[0].kind(),
        PlanExactAliasKind::MayAlias
    );
}

#[test]
fn execution_alias_rejects_partial_and_wrong_input_overlap() {
    for storage in [
        GraphAliasStorage::PartialTarget,
        GraphAliasStorage::ExactWrongInput,
    ] {
        assert!(
            graph_plan_fixture("alias", AliasPolicy::MayAlias { tensor_index: 0 }, storage,)
                .is_err()
        );
    }
}

#[test]
fn execution_alias_rejects_overwrite_before_last_consumer() {
    assert!(graph_plan_fixture(
        "alias_late_consumer",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .is_err());
}

#[test]
fn execution_state_effect_graph_orders_raw_war_waw() {
    let fixture = graph_plan_fixture(
        "state_chain",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    let nodes = fixture.plan.payload().nodes();
    assert_eq!(nodes.len(), 4);
    assert_eq!(nodes[0].state_effects()[0].access(), TensorAccess::Read);
    assert!(nodes[0].dependencies().is_empty());
    assert_eq!(
        nodes[1].state_effects()[0].access(),
        TensorAccess::ReadWrite
    );
    assert_eq!(nodes[1].dependencies(), &[id("node.state-read.0")]);
    assert_eq!(
        nodes[2].state_effects()[0].access(),
        TensorAccess::ReadWrite
    );
    assert_eq!(nodes[2].dependencies(), &[id("node.state-rw.0")]);
    assert_eq!(nodes[3].state_effects()[0].access(), TensorAccess::Read);
    assert_eq!(nodes[3].dependencies(), &[id("node.state-rw.1")]);
}

#[test]
fn execution_state_read_only_nodes_remain_independent() {
    let fixture = graph_plan_fixture(
        "state_read_only",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    assert!(fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .all(
            |node| node.state_effects()[0].access() == TensorAccess::Read
                && node.dependencies().is_empty()
        ));
}

#[test]
fn execution_alias_effect_wire_mutations_are_rejected() {
    let alias = graph_plan_fixture(
        "alias",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    let mut value = serde_json::to_value(&alias.plan).unwrap();
    value["payload"]["nodes"][0]["exact_aliases"] = json!([]);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &alias.family,
        &alias.catalog,
        &alias.policy,
        alias.resolutions.clone(),
    )
    .is_err());

    let state = graph_plan_fixture(
        "state_chain",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    let mut value = serde_json::to_value(&state.plan).unwrap();
    value["payload"]["nodes"][1]["state_effects"][0]["access"] = json!("read");
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &state.family,
        &state.catalog,
        &state.policy,
        state.resolutions.clone(),
    )
    .is_err());

    let mut value = serde_json::to_value(&state.plan).unwrap();
    value["payload"]["nodes"][2]["dependencies"] = json!([]);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &state.family,
        &state.catalog,
        &state.policy,
        state.resolutions,
    )
    .is_err());
}

fn vnext_source_files() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/vnext");
    let mut paths = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn generic_contracts_have_zero_architecture_names() {
    let names = [
        "qwen", "llama", "deepseek", "mistral", "mixtral", "gemma", "chatglm", "internlm",
        "baichuan",
    ];
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap().to_ascii_lowercase();
        for name in names {
            assert!(!source.contains(name), "{} contains {name}", path.display());
        }
    }
}

#[test]
fn silent_success_defaults_are_absent() {
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap();
        assert!(!source.contains("fn unsupported") || !source.contains("Ok(())"));
        assert!(!source.contains("std::env::var"));
        assert!(!source.contains("downcast_ref"));
    }
}

#[test]
fn failure_envelope_wire_limit_precedes_deserialization() {
    let at_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES];
    match FailureEnvelope::decode_untrusted(&at_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(!message.contains("maximum is"));
        }
        other => panic!("equal-to-limit malformed payload hit wrong result: {other:?}"),
    }

    let over_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES + 1];
    match FailureEnvelope::decode_untrusted(&over_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(message.contains("maximum is 8192"));
        }
        other => panic!("oversized payload hit wrong result: {other:?}"),
    }
}
