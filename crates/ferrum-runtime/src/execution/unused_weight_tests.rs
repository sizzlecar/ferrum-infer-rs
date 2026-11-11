//! A phase may keep its complete immutable checkpoint contract while executing
//! only a subgraph. Planning fixtures exercise the real compiler and validators;
//! they make no device-numerical claim.
use crate::test_support::vnext_core_contract::{
    catalog, id, policy, EstimateBehavior, TestConfig, TestFamily, TestPlanningRegistry,
};
use crate::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

struct DormantWeightFamily;

impl ModelFamilyProvider for DormantWeightFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }

    fn validate_config_identity(&self, raw: &Value, config: &TestConfig) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }

    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weights(&self, config: &TestConfig) -> Result<ModelWeights, VNextError> {
        let mut schema = TestFamily.weights(config)?.stored_schema()?.clone();
        schema.components.push(WeightComponentSpec {
            id: id("weight.component.dormant"),
            role: WeightComponentRole::Values,
            external_names: vec!["dormant.bin".into()],
            dimensions: vec![config.width],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.dormant"),
            dimensions: vec![config.width],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: id("weight.component.dormant"),
            },
            required: true,
        });
        Ok(ModelWeights::Stored(schema))
    }

    fn numerical_profiles(
        &self,
        config: &TestConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        TestFamily.numerical_profiles(config)
    }

    fn semantic_program(
        &self,
        config: &TestConfig,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let program = TestFamily.semantic_program(config, profile)?;
        let mut weights = program.weights().to_vec();
        weights.push(WeightReference {
            weight_id: id("weight.dormant"),
            value_id: id("value.weight.dormant"),
            tensor: ProgramTensorSpec {
                dimensions: vec![config.width],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
        ModelProgram::new(
            program.family_id().clone(),
            program.inputs().to_vec(),
            program.blocks().to_vec(),
            program.states().to_vec(),
            weights,
            program.outputs().to_vec(),
        )
    }

    fn semantic_metadata(&self, config: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
}

fn prepared() -> PreparedModelFamily {
    TypedFamilyRegistration::new(DormantWeightFamily)
        .prepare_with_profile(
            &json!({"width": 4}),
            &NumericalProfileId::new("fixture.f32").unwrap(),
        )
        .unwrap()
}

#[test]
fn unused_stored_weight_compiles_without_changing_complete_execution_identity() {
    let family = prepared();
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let plan = compilation.executable().execution_plan();
    let weights = plan.payload().execution_weights().stored().unwrap();
    assert_eq!(weights, &ExecutionWeightPlan::identity(&family).unwrap());
    assert_eq!(weights.schema().components.len(), 2);
    assert!(weights.schema().components.iter().all(|c| c.required));
    assert_eq!(family.program().weights().len(), 2);
    assert!(compilation
        .value_tensors()
        .contains_key(&id("value.weight.dormant")));
    let bound = plan
        .payload()
        .nodes()
        .iter()
        .flat_map(|node| node.values())
        .map(|binding| binding.value_id().clone())
        .collect::<BTreeSet<_>>();
    assert!(bound.contains(&id("value.weight")));
    assert!(!bound.contains(&id("value.weight.dormant")));
    assert_eq!(
        plan.payload()
            .memory()
            .static_allocations()
            .iter()
            .filter(|allocation| allocation.usage() == BufferUsage::Weights)
            .map(ResourceAllocation::size_bytes)
            .sum::<u64>(),
        4 * ElementType::F32.size_bytes(),
        "a declaration alone cannot invent an active allocation or source upload"
    );
}

#[test]
fn unused_stored_weight_does_not_excuse_missing_active_semantic_bindings() {
    let family = prepared();
    let complete = ["value.input", "value.weight", "value.state", "value.output"]
        .into_iter()
        .map(id)
        .collect::<BTreeSet<ProgramValueId>>();
    ExecutionPlan::validate_semantic_coverage(&family, &complete).unwrap();
    for missing in &complete {
        let mut incomplete = complete.clone();
        incomplete.remove(missing);
        assert!(
            ExecutionPlan::validate_semantic_coverage(&family, &incomplete).is_err(),
            "the consumed weight, input, state and output still require bindings: {missing}"
        );
    }
    let program = family.program();
    let incomplete = ModelProgram::new(
        program.family_id().clone(),
        program.inputs().to_vec(),
        program.blocks().to_vec(),
        program.states().to_vec(),
        program
            .weights()
            .iter()
            .filter(|weight| weight.weight_id != id("weight.dormant"))
            .cloned()
            .collect(),
        program.outputs().to_vec(),
    )
    .unwrap();
    assert!(family
        .stored_weight_schema()
        .unwrap()
        .validate_program_references(family.family_id(), &incomplete)
        .is_err());
}
