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
    assert_eq!(resolved_weight.format_id(), &id("weight-format.dense"));
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
        weight.weight().unwrap().format_id(),
        &id("weight-format.safetensors.mixed-gptq")
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
