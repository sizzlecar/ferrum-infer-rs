mod vnext_core_contract;

use vnext_core_contract::*;

const NOVEL_OPERATION_ID: &str = "operation.extension.scale_bias";

struct EchoReferenceOracle {
    descriptor: OperationOracleDescriptor,
    calls: Arc<AtomicUsize>,
}

impl OperationOracle for EchoReferenceOracle {
    fn descriptor(&self) -> &OperationOracleDescriptor {
        &self.descriptor
    }

    fn invoke(
        &self,
        request: &OperationOracleRequest,
    ) -> Result<OperationOracleResult, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        OperationOracleResult::new(vec![request.inputs()[0].clone()])
    }
}

struct SyntheticExtensionFamily {
    family_id: ModelFamilyId,
    metadata_id: ExternalModelMetadataId,
    operation_id: OperationId,
    required_version: ContractVersion,
    recurrent: bool,
}

impl SyntheticExtensionFamily {
    fn dense(operation_id: &str, required_version: ContractVersion) -> Self {
        Self {
            family_id: id("family.extension.dense"),
            metadata_id: id("metadata.extension.dense"),
            operation_id: id(operation_id),
            required_version,
            recurrent: false,
        }
    }

    fn recurrent(operation_id: &str) -> Self {
        Self {
            family_id: id("family.extension.recurrent"),
            metadata_id: id("metadata.extension.recurrent"),
            operation_id: id(operation_id),
            required_version: ContractVersion::new(1, 0),
            recurrent: true,
        }
    }
}

impl ModelFamilyProvider for SyntheticExtensionFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata_id.clone()])
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
        Ok(self.metadata_id.clone())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        let mut schema = TestFamily.weight_schema(config)?;
        if !self.recurrent {
            schema.components.push(WeightComponentSpec {
                id: id("weight.component.mask"),
                role: WeightComponentRole::Values,
                external_names: vec!["mask.bin".to_owned()],
                dimensions: vec![config.width],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            });
            schema.tensors.push(WeightTensorSpec {
                id: id("weight.mask"),
                dimensions: vec![config.width],
                logical_element_type: ElementType::U8,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.mask"),
                },
                required: true,
            });
        }
        Ok(schema)
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let auxiliary_value: ProgramValueId = if self.recurrent {
            id("value.state")
        } else {
            id("value.mask")
        };
        let states = self
            .recurrent
            .then(|| StateSpec {
                id: id("state.recurrent"),
                value_id: auxiliary_value.clone(),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
                initialization: StateInitialization::Zero,
            })
            .into_iter()
            .collect();
        let mut weights = vec![WeightReference {
            weight_id: id("weight.matrix"),
            value_id: id("value.weight"),
            tensor: ProgramTensorSpec {
                dimensions: vec![config.width],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        }];
        if !self.recurrent {
            weights.push(WeightReference {
                weight_id: id("weight.mask"),
                value_id: auxiliary_value.clone(),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
        }
        ModelProgram::new(
            self.family_id.clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.extension".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.extension"),
                    operation_id: self.operation_id.clone(),
                    required_version: self.required_version,
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id("value.input"), id("value.weight"), auxiliary_value],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            states,
            weights,
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

fn prepare(provider: SyntheticExtensionFamily) -> PreparedModelFamily {
    TypedFamilyRegistration::new(provider)
        .prepare(&json!({"width": 4}))
        .unwrap()
}

fn compile(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    registry: &TestPlanningRegistry,
) -> Result<ProgramPlanCompilation, VNextError> {
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))?;
    let planning = registry.planning();
    ProgramPlanCompiler::compile(family, catalog, &policy(4096), &planning, &options)
}

fn extension_catalog(
    operations: Vec<OperationDescriptor>,
    device_id: &str,
    class: DeviceClass,
) -> CapabilityCatalog {
    let device_id: DeviceId = id(device_id);
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let class_name = match class {
        DeviceClass::Host => "host",
        DeviceClass::Accelerator => "accelerator",
        DeviceClass::Reference => "reference",
    };
    let providers = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| {
            (
                operation.id.clone(),
                vec![OperationProviderDescriptor::new(
                    id(format!("provider.extension.{class_name}.{index}")),
                    operation.id.clone(),
                    operation.fingerprint().unwrap(),
                    sha('f'),
                    operation.version,
                    device_id.clone(),
                    capabilities.clone(),
                    BTreeSet::from([id("weight-format.dense")]),
                    BTreeSet::new(),
                    contiguous_storage_bindings(operation),
                    format!("resource-estimator.extension.{class_name}.{index}"),
                    ContractVersion::new(1, 0),
                    sha('e'),
                )
                .unwrap()],
            )
        })
        .collect();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: match class {
                DeviceClass::Host => sha('a'),
                DeviceClass::Accelerator => sha('b'),
                DeviceClass::Reference => sha('c'),
            },
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        operations,
        providers,
        vec![EngineProviderDescriptor::new(
            id(format!("provider.engine.extension.{class_name}")),
            ContractVersion::new(1, 0),
            sha('8'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

fn novel_operation() -> OperationDescriptor {
    let reference = operation();
    let mut novel = reference.clone();
    novel.id = id(NOVEL_OPERATION_ID);
    novel.oracle = OracleSpec::ReferenceOperation {
        operation_id: reference.id,
        version: ContractVersion::new(1, 0),
    };
    novel.validate().unwrap();
    novel
}

fn oracle_tensor(element_type: ElementType, bytes: Vec<u8>) -> OracleTensor {
    OracleTensor::new(vec![4], element_type, bytes).unwrap()
}

#[test]
fn existing_operation_dense_family_is_additive() {
    let family = prepare(SyntheticExtensionFamily::dense(
        "operation.main",
        ContractVersion::new(1, 0),
    ));
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let compilation = compile(&family, &catalog, &registry).unwrap();
    let node = &compilation.executable().execution_plan().payload().nodes()[0];

    assert!(family.program().states().is_empty());
    assert!(node.state_effects().is_empty());
    assert_eq!(node.operation_id().as_str(), "operation.main");
    assert_eq!(
        node.selection().selected_provider().as_str(),
        "provider.operation.reference"
    );
}

#[test]
fn recurrent_family_reuses_sequence_state_contract() {
    let family = prepare(SyntheticExtensionFamily::recurrent("operation.main"));
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let compilation = compile(&family, &catalog, &registry).unwrap();
    let node = &compilation.executable().execution_plan().payload().nodes()[0];

    assert_eq!(family.program().states().len(), 1);
    assert_eq!(
        family.program().states()[0].lifetime,
        StateLifetime::Sequence
    );
    assert_eq!(node.state_effects().len(), 1);
    assert_eq!(
        node.state_effects()[0].state_id().as_str(),
        "state.recurrent"
    );
}

#[test]
fn novel_operation_is_additive_to_catalog_provider_and_oracle_graph() {
    let reference = operation();
    let novel = novel_operation();
    let catalog = extension_catalog(
        vec![reference.clone(), novel.clone()],
        "device.reference.extension.0",
        DeviceClass::Reference,
    );
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let family = prepare(SyntheticExtensionFamily::dense(
        NOVEL_OPERATION_ID,
        ContractVersion::new(1, 0),
    ));
    let compilation = compile(&family, &catalog, &registry).unwrap();
    let node = &compilation.executable().execution_plan().payload().nodes()[0];

    assert_eq!(node.operation_id(), &novel.id);
    assert_eq!(node.operation_version(), ContractVersion::new(1, 0));
    assert!(registry.contract_calls.load(Ordering::SeqCst) >= 1);
    assert!(registry.estimator_calls.load(Ordering::SeqCst) >= 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let oracle_descriptor = OperationOracleDescriptor::new(
        id("oracle.extension.reference"),
        ContractVersion::new(1, 0),
        sha('9'),
        reference.id.clone(),
        reference.fingerprint().unwrap(),
    )
    .unwrap();
    let oracle_registry = OperationOracleRegistry::new(
        &catalog,
        vec![
            Box::new(TestOperationContract {
                descriptor: reference.clone(),
                calls: Arc::new(AtomicUsize::new(0)),
                reject_signature: false,
            }),
            Box::new(TestOperationContract {
                descriptor: novel.clone(),
                calls: Arc::new(AtomicUsize::new(0)),
                reject_signature: false,
            }),
        ],
        vec![OperationOracleRegistration::new(
            oracle_descriptor.clone(),
            Box::new(EchoReferenceOracle {
                descriptor: oracle_descriptor,
                calls: calls.clone(),
            }),
        )
        .unwrap()],
    )
    .unwrap();
    let bound = oracle_registry.bind(&novel.id).unwrap();
    assert_eq!(bound.requested_operation_id(), &novel.id);
    assert_eq!(bound.terminal_operation_id(), &reference.id);
    let f32_bytes = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let result = bound
        .invoke(
            vec![
                oracle_tensor(ElementType::F32, f32_bytes.clone()),
                oracle_tensor(ElementType::F32, vec![0; 16]),
                oracle_tensor(ElementType::U8, vec![0; 4]),
            ],
            BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(result.outputs()[0].bytes(), f32_bytes);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn reference_backend_reuses_the_same_prepared_model_program() {
    let family = prepare(SyntheticExtensionFamily::dense(
        "operation.main",
        ContractVersion::new(1, 0),
    ));
    let program_fingerprint = family.program().fingerprint().unwrap();
    let accelerator = extension_catalog(
        vec![operation()],
        "device.accelerator.extension.0",
        DeviceClass::Accelerator,
    );
    let reference = extension_catalog(
        vec![operation()],
        "device.reference.extension.0",
        DeviceClass::Reference,
    );
    let accelerator_registry =
        TestPlanningRegistry::new(&accelerator, 64, 32, EstimateBehavior::Correct);
    let reference_registry =
        TestPlanningRegistry::new(&reference, 64, 32, EstimateBehavior::Correct);

    let accelerator_plan = compile(&family, &accelerator, &accelerator_registry).unwrap();
    let reference_plan = compile(&family, &reference, &reference_registry).unwrap();

    assert_eq!(family.program().fingerprint().unwrap(), program_fingerprint);
    assert_ne!(
        accelerator_plan.executable().execution_plan().plan_hash(),
        reference_plan.executable().execution_plan().plan_hash()
    );
    assert_eq!(
        accelerator_plan
            .executable()
            .execution_plan()
            .payload()
            .nodes()[0]
            .operation_id(),
        reference_plan
            .executable()
            .execution_plan()
            .payload()
            .nodes()[0]
            .operation_id()
    );
}

#[test]
fn unsupported_backend_fails_with_missing_operation_or_version_before_estimation() {
    let missing_family = prepare(SyntheticExtensionFamily::dense(
        NOVEL_OPERATION_ID,
        ContractVersion::new(1, 0),
    ));
    let missing_catalog = extension_catalog(
        vec![operation()],
        "device.reference.extension.0",
        DeviceClass::Reference,
    );
    let missing_registry =
        TestPlanningRegistry::new(&missing_catalog, 64, 32, EstimateBehavior::Correct);
    match compile(&missing_family, &missing_catalog, &missing_registry).unwrap_err() {
        VNextError::UnsupportedOperation {
            node_id,
            operation_id,
            device_id,
            ..
        } => {
            assert_eq!(node_id.as_deref(), Some("node.extension"));
            assert_eq!(operation_id, NOVEL_OPERATION_ID);
            assert_eq!(device_id, "device.reference.extension.0");
        }
        error => panic!("expected structured missing-operation error, got {error}"),
    }
    assert_eq!(missing_registry.estimator_calls.load(Ordering::SeqCst), 0);

    let version_family = prepare(SyntheticExtensionFamily::dense(
        NOVEL_OPERATION_ID,
        ContractVersion::new(2, 0),
    ));
    let version_catalog = extension_catalog(
        vec![operation(), novel_operation()],
        "device.reference.extension.0",
        DeviceClass::Reference,
    );
    let version_registry =
        TestPlanningRegistry::new(&version_catalog, 64, 32, EstimateBehavior::Correct);
    match compile(&version_family, &version_catalog, &version_registry).unwrap_err() {
        VNextError::IncompatibleOperationVersion {
            node_id,
            operation_id,
            required_major,
            required_minor,
            available_major,
            available_minor,
        } => {
            assert_eq!(node_id.as_deref(), Some("node.extension"));
            assert_eq!(operation_id, NOVEL_OPERATION_ID);
            assert_eq!((required_major, required_minor), (2, 0));
            assert_eq!((available_major, available_minor), (1, 0));
        }
        error => panic!("expected structured version error, got {error}"),
    }
    assert_eq!(version_registry.estimator_calls.load(Ordering::SeqCst), 0);
}
