mod vnext_core_contract;

use vnext_core_contract::*;

fn prepared_with_weight_schema(schema: WeightSchema) -> PreparedModelFamily {
    TypedFamilyRegistration::new(FixedSchemaFamily { schema })
        .prepare(&json!({"width": 4}))
        .unwrap()
}

fn compile_options() -> ProgramPlanCompileOptions {
    ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap()
}

#[test]
fn determinism_compile_option_retains_every_operation_output() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut options = compile_options();
    options.retain_all_outputs_for_determinism(&family).unwrap();
    let expected = family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .flat_map(|node| node.outputs.iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(options.completion_retention().values(), &expected);

    let planning = registry.planning();
    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy(4096), &planning, &options)
            .unwrap();
    let retained = compilation
        .executable()
        .execution_plan()
        .payload()
        .retained_completion_values()
        .iter()
        .map(|value| value.value_id().clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(retained, expected);
}

struct PaddedDenseMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl PaddedDenseMaterializer {
    fn new() -> Self {
        Self::with_implementation_fingerprint('9')
    }

    fn with_implementation_fingerprint(fingerprint_byte: char) -> Self {
        Self::with_fidelity(fingerprint_byte, WeightMaterializationFidelity::Exact)
    }

    fn with_fidelity(fingerprint_byte: char, fidelity: WeightMaterializationFidelity) -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.padded-dense"),
                ContractVersion::new(1, 0),
                sha(fingerprint_byte),
                fidelity,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

impl WeightMaterializer for PaddedDenseMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.layout_id = id("weight-layout.test.padded-dense");
        schema.components[0].dimensions = vec![8];
        schema.tensors[0].physical_layout = PhysicalWeightLayout::Stored {
            component: PhysicalWeightComponentBinding {
                component_id: id("weight.component"),
                storage: PhysicalStorageLayout::Contiguous {
                    padding: PhysicalWeightPadding::ZeroFill {
                        padded_dimensions: vec![8],
                    },
                },
            },
        };
        Ok(schema)
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [source_component] = source_components else {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "padded test materializer requires one source component".to_owned(),
            });
        };
        let source_payload = source.component(source_component)?;
        let output_bytes =
            usize::try_from(execution_component.physical_bytes()?).map_err(|_| {
                VNextError::InvalidExecutionPlan {
                    reason: "padded test component exceeds host address space".to_owned(),
                }
            })?;
        let mut bytes = source_payload.bytes().to_vec();
        bytes.resize(output_bytes, 0);
        WeightComponentPayload::from_ordered_sources(
            execution_component,
            execution_component.external_names.clone(),
            source_payload.source_files().to_vec(),
            execution_component.dimensions.clone(),
            execution_component.physical_element_type(),
            bytes,
        )
    }
}

struct LogicalMutationMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl LogicalMutationMaterializer {
    fn new() -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.logical-mutation"),
                ContractVersion::new(1, 0),
                sha('7'),
                WeightMaterializationFidelity::Exact,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

impl WeightMaterializer for LogicalMutationMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.components[0].dimensions = vec![5];
        schema.tensors[0].dimensions = vec![5];
        Ok(schema)
    }

    fn materialize_component<'source>(
        &self,
        _source: &'source dyn WeightComponentSource,
        _source_components: &[&WeightComponentSpec],
        _execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        Err(VNextError::InvalidExecutionPlan {
            reason: "logically invalid test materializer must never initialize weights".to_owned(),
        })
    }
}

#[test]
fn semantic_program_compiles_through_the_registered_provider_authority() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let plan = compilation.executable().execution_plan();
    assert_eq!(plan.payload().nodes().len(), 1);
    assert!(plan.payload().retained_completion_values().is_empty());
    assert_eq!(
        plan.payload().execution_weights().schema(),
        family.weight_schema()
    );
    assert_eq!(
        plan.payload()
            .execution_weights()
            .source_schema_fingerprint(),
        family.weight_schema().fingerprint().unwrap()
    );
    assert_eq!(
        plan.payload()
            .execution_weights()
            .materializer_id()
            .as_str(),
        "weight-materializer.identity"
    );
    assert_eq!(compilation.node_resolutions().len(), 1);
    assert_eq!(
        compilation
            .value_tensors()
            .get(&id("value.output"))
            .unwrap()
            .dimensions(),
        &[4]
    );

    let node = &plan.payload().nodes()[0];
    let weight = node
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    let resolved_weight = weight
        .weight()
        .expect("provider binding must retain the physical weight contract");
    assert_eq!(resolved_weight.weight_id(), &id("weight.matrix"));
    assert_eq!(
        serde_json::to_value(resolved_weight).unwrap()["format_id"],
        json!("weight-format.dense")
    );
    assert_eq!(resolved_weight.layout_id(), &id("weight-layout.dense"));
    assert_eq!(resolved_weight.components().len(), 1);
    assert_eq!(resolved_weight.components()[0].physical_dimensions(), &[4]);
    assert_eq!(weight.storage().components().len(), 1);
    let component = &weight.storage().components()[0];
    assert!(component
        .resource_id()
        .as_str()
        .starts_with("resource/weight-arena/sha256/"));
    assert_eq!(
        component.offset_bytes() % node.provider_resources().value_alignment_bytes(),
        0
    );
    assert!(registry.estimator_calls.load(Ordering::SeqCst) >= 2);
}

#[test]
fn approximate_materializer_is_not_authorized_by_capability_registration() {
    let family = TestRegistry::new().prepare();
    let materializer_id: WeightMaterializerId = id("weight-materializer.test.padded-dense");
    let materializers = WeightMaterializerRegistry::new(vec![Box::new(
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate),
    )])
    .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut options = compile_options();
    options.require_weight_materializer(materializer_id.clone());

    assert_eq!(
        catalog
            .weight_materializer(&materializer_id)
            .unwrap()
            .fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    assert_eq!(
        serde_json::to_value(catalog.weight_materializer(&materializer_id).unwrap()).unwrap()
            ["fidelity"],
        json!("approximate")
    );
    let error = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires explicit numerical-quality approval"),
        "{error}"
    );
}

#[test]
fn trusted_materializer_changes_physical_plan_memory_and_wire_requires_its_witness() {
    let family = TestRegistry::new().prepare();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer::new())]).unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let identity = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy,
        &registry.planning(),
        &compile_options(),
    )
    .unwrap();
    let mut options = compile_options();
    let materializer_id: WeightMaterializerId = id("weight-materializer.test.padded-dense");
    options.require_weight_materializer(materializer_id.clone());
    let transformed = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy,
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap();
    let plan = transformed.executable().execution_plan();
    assert_eq!(
        plan.payload().execution_weights().materializer_id(),
        &materializer_id
    );
    assert_eq!(
        plan.payload().execution_weights().schema().layout_id,
        id("weight-layout.test.padded-dense")
    );
    assert_eq!(
        plan.payload().memory().static_bytes(),
        identity
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .static_bytes()
            + 16
    );
    plan.validate_against(&family, &catalog, &policy, transformed.node_resolutions())
        .unwrap();

    let wire = plan.to_json().unwrap();
    assert!(ExecutionPlan::from_json_validated(
        &wire,
        &family,
        &catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .is_err());
    let trusted_weights = materializers
        .select_exact(&family, &catalog, &materializer_id)
        .unwrap();
    let mismatched_materializers = WeightMaterializerRegistry::new(vec![Box::new(
        PaddedDenseMaterializer::with_implementation_fingerprint('6'),
    )])
    .unwrap();
    let mismatched_catalog = mismatched_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &mismatched_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let capability_mutated_materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer {
            descriptor: WeightMaterializerDescriptor::new(
                materializer_id.clone(),
                ContractVersion::new(1, 0),
                sha('9'),
                WeightMaterializationFidelity::Exact,
                BTreeSet::new(),
            )
            .unwrap(),
        })])
        .unwrap();
    let capability_mutated_catalog = capability_mutated_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &capability_mutated_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let fidelity_mutated_materializers = WeightMaterializerRegistry::new(vec![Box::new(
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate),
    )])
    .unwrap();
    let fidelity_mutated_catalog = fidelity_mutated_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &fidelity_mutated_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let restored = ExecutionPlan::from_json_validated_with_execution_weights(
        &wire,
        &family,
        &catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
        CompletionRetentionSpec::default(),
        trusted_weights,
    )
    .unwrap();
    assert_eq!(restored, *plan);
}

#[test]
fn materializer_cannot_change_the_prepared_logical_weight_contract() {
    let family = TestRegistry::new().prepare();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(LogicalMutationMaterializer::new())])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let error = materializers
        .select_exact(
            &family,
            &catalog,
            &id("weight-materializer.test.logical-mutation"),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changes the prepared family's logical tensor contract"),
        "{error}"
    );
}

#[test]
fn dense_binding_in_mixed_checkpoint_does_not_require_the_container_format() {
    let mut schema = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    schema.format_id = id("weight-format.safetensors.mixed-gptq");
    schema.layout_id = id("weight-layout.synthetic.mixed-gptq");
    let family = prepared_with_weight_schema(schema);
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &compile_options(),
    )
    .unwrap();
    let weight = compilation.executable().execution_plan().payload().nodes()[0]
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    assert_eq!(
        serde_json::to_value(weight.weight().unwrap()).unwrap()["format_id"],
        json!("weight-format.safetensors.mixed-gptq")
    );
}

#[test]
fn quantized_binding_still_requires_its_format_and_abi_before_allocation() {
    let schema = WeightSchema {
        format_id: id("weight-format.synthetic.quantized"),
        layout_id: id("weight-layout.synthetic.quantized"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("weight.component"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["weight.bin".to_owned()],
            dimensions: vec![1],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id("quantization.synthetic.int4"),
                logical_values_per_block: 4,
                bytes_per_block: 2,
            }),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.matrix"),
            dimensions: vec![4],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(id("weight.component")),
                block_axis: 0,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    };
    let family = prepared_with_weight_schema(schema);
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let error = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &compile_options(),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("weight-format.synthetic.quantized"),
        "{message}"
    );
    assert!(message.contains("quantization.synthetic.int4"), "{message}");
}

#[test]
fn completion_retention_binds_one_typed_output_and_requires_expected_wire_policy() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    assert!(options.retain_completion_value(id("value.output")));

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let plan = compilation.executable().execution_plan();
    let checkpoint = plan.completion_checkpoint(&id("value.output")).unwrap();
    assert_eq!(checkpoint.producer_node_id(), &id("node.main"));
    assert_eq!(checkpoint.output_ordinal(), 0);
    assert_eq!(checkpoint.tensor().dimensions(), &[4]);
    assert_eq!(
        plan.payload().terminal_output_resources(),
        std::slice::from_ref(checkpoint.resource_id())
    );
    let output_descriptor = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id() == checkpoint.resource_id())
        .unwrap();
    assert_eq!(output_descriptor.lifetime(), AllocationLifetime::Step);
    assert!(matches!(
        output_descriptor.demand(),
        DynamicResourceDemand::ActualSequences {
            bytes_per_sequence: 16,
            maximum_sequences: 3,
        }
    ));
    let readback = checkpoint
        .readback_request(3, HostTransferLayout::new(ElementType::F32, 4).unwrap())
        .unwrap();
    assert_eq!(readback.node_id(), checkpoint.producer_node_id());
    assert_eq!(readback.resource_id(), checkpoint.resource_id());
    assert_eq!(readback.participant_index(), 3);
    let work_readback = plan
        .completion_checkpoint_readback_for_work(&id("value.output"), 3, &resource_work(&[2]))
        .unwrap();
    assert_eq!(work_readback.output_layout().element_count(), 4);
    assert!(checkpoint
        .readback_request(0, HostTransferLayout::new(ElementType::U8, 4).unwrap())
        .unwrap_err()
        .to_string()
        .contains("element type differs"));
    assert!(checkpoint
        .readback_request(0, HostTransferLayout::new(ElementType::F32, 5).unwrap())
        .unwrap_err()
        .to_string()
        .contains("exceeds retained activation capacity"));

    let wire = plan.to_json().unwrap();
    assert!(ExecutionPlan::from_json_validated(
        &wire,
        &family,
        &catalog,
        &policy,
        compilation.node_resolutions().to_vec(),
    )
    .is_err());
    let restored = ExecutionPlan::from_json_validated_with_completion_retention(
        &wire,
        &family,
        &catalog,
        &policy,
        compilation.node_resolutions().to_vec(),
        CompletionRetentionSpec::new(BTreeSet::from([id("value.output")])),
    )
    .unwrap();
    assert_eq!(restored, *plan);

    let mut forged = serde_json::from_slice::<Value>(&wire).unwrap();
    forged["payload"]["retained_completion_values"][0]["resource_id"] =
        json!("resource/forged-retained-output");
    rehash_plan_json(&mut forged);
    let forged = serde_json::to_vec(&forged).unwrap();
    assert!(
        ExecutionPlan::from_json_validated_with_completion_retention(
            &forged,
            &family,
            &catalog,
            &policy,
            compilation.node_resolutions().to_vec(),
            CompletionRetentionSpec::new(BTreeSet::from([id("value.output")])),
        )
        .is_err()
    );
}

#[test]
fn completion_retention_rejects_inputs_weights_and_unknown_values_before_planning() {
    for value_id in ["value.input", "value.weight", "value.unknown"] {
        let family = TestRegistry::new().prepare();
        let catalog = catalog();
        let policy = policy(4096);
        let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
        let planning = registry.planning();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.input"),
            ProgramTensorSpec {
                dimensions: vec![4],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.retain_completion_value(id(value_id));

        let error = ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("completion retention must reference a semantic node output"),
            "{value_id}: {error}"
        );
    }
}

#[test]
fn compilation_rejects_missing_or_guessed_product_input_capacity() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::new()).unwrap();
    let error =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap_err();
    assert!(error
        .to_string()
        .contains("every program input requires an explicit canonical tensor capacity"));
}

#[test]
fn compilation_reports_the_exact_tensor_binding_on_signature_mismatch() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::U32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let error =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("input[0] `value.input`"), "{message}");
    assert!(message.contains("dtype=U32"), "{message}");
}

#[test]
fn weight_arena_reaches_provider_alignment_fixed_point() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::ArenaAlignment64);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let node = &compilation.executable().execution_plan().payload().nodes()[0];
    assert_eq!(node.provider_resources().value_alignment_bytes(), 64);
    assert!(registry.estimator_calls.load(Ordering::SeqCst) >= 3);
    let weight = node
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    assert!(weight
        .storage()
        .components()
        .iter()
        .all(|component| component.offset_bytes() % 64 == 0));
}
