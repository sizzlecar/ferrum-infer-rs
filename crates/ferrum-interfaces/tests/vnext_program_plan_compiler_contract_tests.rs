mod vnext_core_contract;

use vnext_core_contract::*;

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
