use super::*;
use serde_json::Value;
mod managed;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CompositionConfig {
    stages: usize,
}

#[test]
fn prepared_composition_preserves_definition_and_compile_geometry() {
    let built = joined(shared_chain(), 8).unwrap();
    let schema = built.schema.clone();
    let program = built.program.clone();
    let profile = built.profile.clone();
    let options = built.options.clone();
    let (family, actual_options) = built
        .prepare_family(&CompositionConfig { stages: 2 }, id("metadata.joined.test"))
        .unwrap();
    assert_eq!(family.weights(), &ModelWeights::Stored(schema));
    assert_eq!(family.program(), &program);
    assert_eq!(family.numerical_profile(), &profile);
    assert_eq!(actual_options, options);
}

#[test]
fn prepared_composition_still_requires_core_schema_and_numerical_validation() {
    let config = CompositionConfig { stages: 2 };
    let mut built = joined(shared_chain(), 8).unwrap();
    built.schema.components.clear();
    assert!(built
        .prepare_family(&config, id("metadata.joined.test"))
        .is_err());
    let mut built = joined(shared_chain(), 8).unwrap();
    built.profile.operations.clear();
    assert!(built
        .prepare_family(&config, id("metadata.joined.test"))
        .is_err());
}

const PROFILE: &str = "fragment.test.f32.v1";
fn id<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.to_owned()).unwrap()
}
fn tensor() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![1, 2],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn geometry() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    BTreeMap::from([
        (id("value.input"), tensor()),
        (id("value.output"), tensor()),
    ])
}

fn joined(fragments: Vec<Fragment>, capacity: u64) -> Result<StoredComposition, VNextError> {
    joined_with_node_capacity(fragments, capacity, 8)
}
fn joined_with_node_capacity(
    fragments: Vec<Fragment>,
    capacity: u64,
    maximum_nodes: usize,
) -> Result<StoredComposition, VNextError> {
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id("family.joined.test"),
            profile: id("joined.test.f32.v1"),
            format: id("weight-format.safetensors.dense"),
            layout: id("weight-layout.joined.test"),
            block: "block.joined".into(),
        },
        fragments,
        vec![id("value.input")],
        vec![id("value.output.second")],
        id("value.output.second"),
        StoredCompositionLimits {
            maximum_nodes,
            maximum_stored_bytes: capacity,
            maximum_metadata_bytes: 16384,
        },
    )
}
#[test]
fn node_capacity_is_typed_and_does_not_change_valid_composition() {
    assert!(joined_with_node_capacity(shared_chain(), 8, 2).is_ok());
    assert!(matches!(
        joined_with_node_capacity(shared_chain(), 8, 1),
        Err(VNextError::ModelProgramNodeCapacityExceeded {
            required_nodes: 2,
            maximum_nodes: 1,
        })
    ));
    let mut invalid = shared_chain();
    invalid[1].schema.as_mut().unwrap().format_id = id("weight-format.other");
    assert!(!matches!(
        joined_with_node_capacity(invalid, 8, 2),
        Ok(_) | Err(VNextError::ModelProgramNodeCapacityExceeded { .. })
    ));
}
fn shared_chain() -> Vec<Fragment> {
    let make = |instance, input| {
        Fragment::family_instance(
            &Family::new(),
            &false,
            PROFILE,
            &geometry(),
            mapping(&[("value.input", input)]).unwrap(),
            Some(instance),
        )
        .unwrap()
    };
    vec![
        make("first", "value.input"),
        make("second", "value.output.first"),
    ]
}
#[test]
fn composition_shares_identical_source_and_weight_value_across_uses_with_exact_capacity() {
    let graph = joined(shared_chain(), 8).unwrap();
    assert_eq!(graph.schema.components.len(), 1);
    assert_eq!(graph.schema.tensors.len(), 1);
    assert_eq!(graph.program.weights().len(), 1);
    assert_eq!(
        graph.program.blocks()[0].nodes[0].inputs[1],
        graph.program.blocks()[0].nodes[1].inputs[1]
    );
    assert_eq!(
        graph.program.blocks()[0].nodes[1].inputs[0],
        id("value.output.first")
    );
    assert_eq!(graph.schema.components[0].physical_bytes().unwrap(), 8);
    assert!(joined(shared_chain(), 7).is_err());
}
#[test]
fn composition_rejects_conflicting_source_geometry_producers_and_arithmetic() {
    for mutation in 0..6 {
        let mut fragments = shared_chain();
        match mutation {
            0 => {
                fragments[1].schema.as_mut().unwrap().components[0].external_names =
                    vec!["different.source".into()]
            }
            1 => {
                fragments[1]
                    .specs
                    .get_mut(&id::<ProgramValueId>("value.output.first"))
                    .unwrap()
                    .dimensions = vec![2, 1]
            }
            2 => {
                fragments[1]
                    .boundaries
                    .insert(id("value.output.first"), ElementType::F32);
            }
            3 => fragments[1].operations[0].multiplication_type = Some(ElementType::Bf16),
            4 => fragments[1].schema.as_mut().unwrap().format_id = id("weight-format.other"),
            5 => fragments[1].nodes[0].id = id("node.add.first"),
            _ => unreachable!(),
        }
        assert!(joined(fragments, 16).is_err(), "mutation {mutation}");
    }
}

/// A real stateless stored-weight family: the fragment must preserve its
/// physical component while changing every logical definition consistently.
struct Family(ModelFamilyId, bool);
impl Family {
    fn new() -> Self {
        Self(id("family.fragment.test"), false)
    }
}
impl ModelFamilyProvider for Family {
    // Whether checkpoint input roles are explicitly declared.
    type Config = bool;
    fn family_id(&self) -> &ModelFamilyId {
        &self.0
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fragment.test")])
    }
    fn validate_config_identity(&self, raw: &Value, c: &bool) -> Result<(), VNextError> {
        if raw.as_bool() == Some(*c) {
            Ok(())
        } else {
            Err(invalid("noncanonical fixture"))
        }
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &bool,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fragment.test"))
    }
    fn parse_config(&self, raw: &Value) -> Result<bool, VNextError> {
        raw.as_bool().ok_or_else(|| invalid("expected bool"))
    }
    fn weights(&self, _: &bool) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fragment.test"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.original"),
                role: WeightComponentRole::Values,
                external_names: vec!["checkpoint.weight".to_owned()],
                dimensions: vec![1, 2],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.logical"),
                dimensions: vec![1, 2],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.original"),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &bool) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(PROFILE).unwrap(),
            family_id: self.0.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id("value.output"),
            boundaries: BTreeMap::from([(id("value.output"), ElementType::F32)]),
            states: vec![],
            kv_storage: vec![],
            operations: vec![NumericalOperationContract {
                operation_id: id(PLAIN_RESIDUAL_ADD_F32_OPERATION_ID),
                version: ContractVersion::new(1, 0),
                multiplication_type: None,
                accumulation_type: None,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }],
        };
        FamilyNumericalProfiles::new(
            &self.0,
            ContractVersion::new(1, 0),
            vec![profile.clone()],
            vec![profile.id],
        )
    }
    fn specialize_weight_schema(
        &self,
        _: &bool,
        source: &WeightSchema,
        _: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        let mut specialized = source.clone();
        if self.1 {
            specialized.tensors[0].logical_element_type = ElementType::Bf16;
        }
        Ok(specialized)
    }
    fn semantic_program(
        &self,
        checkpoint: &bool,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let program = ModelProgram::new(
            self.0.clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.add"),
                    operation_id: id(PLAIN_RESIDUAL_ADD_F32_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id("value.input"), id("value.weight")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.logical"),
                value_id: id("value.weight"),
                tensor: tensor(),
            }],
            vec![id("value.output")],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id("value.input"), tensor())]))?;
        if *checkpoint {
            program.with_checkpoint_inputs(ProgramCheckpointInputs::new(
                id("value.input"),
                BTreeSet::new(),
            )?)
        } else {
            Ok(program)
        }
    }
    fn semantic_metadata(&self, _: &bool) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[test]
fn renames_logical_definitions_and_preserves_physical_weight_identity() {
    let family = Family::new();
    let fragment = Fragment::family_instance(
        &family,
        &false,
        PROFILE,
        &geometry(),
        mapping(&[("value.input", "value.joined")]).unwrap(),
        Some("video"),
    )
    .unwrap();
    assert_eq!(fragment.nodes[0].id, id("node.add.video"));
    assert_eq!(
        fragment.nodes[0].inputs,
        vec![id("value.joined"), id("value.weight.video")]
    );
    assert_eq!(fragment.nodes[0].outputs, vec![id("value.output.video")]);
    assert_eq!(fragment.weights[0].weight_id, id("weight.logical.video"));
    assert_eq!(fragment.weights[0].value_id, id("value.weight.video"));
    let schema = fragment.schema.unwrap();
    assert_eq!(schema.components[0].id, id("component.original"));
    assert_eq!(schema.tensors[0].id, id("weight.logical.video"));
    assert_eq!(
        schema.tensors[0].physical_layout,
        PhysicalWeightLayout::Dense {
            component_id: id("component.original")
        }
    );
    assert_eq!(
        fragment.boundaries,
        BTreeMap::from([(id("value.output.video"), ElementType::F32)])
    );
    assert_eq!(
        fragment.specs.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from([id("value.joined"), id("value.output.video")])
    );
}

#[test]
fn rejects_checkpoint_semantics_instead_of_dropping_them() {
    let error = Fragment::family(
        &Family::new(),
        &true,
        PROFILE,
        &geometry(),
        mapping(&[("value.input", "value.joined")]).unwrap(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("checkpoint input roles"));
}

#[test]
fn rejects_unsupported_schema_specialization_instead_of_skipping_it() {
    let mut family = Family::new();
    family.1 = true;
    let error = Fragment::family(
        &family,
        &false,
        PROFILE,
        &geometry(),
        mapping(&[("value.input", "value.joined")]).unwrap(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("schema specialization"));
}

#[test]
fn rejects_incomplete_changed_or_aliased_geometry() {
    let family = Family::new();
    let call = |geometry: &BTreeMap<ProgramValueId, ProgramTensorSpec>, target: &str| {
        Fragment::family_instance(
            &family,
            &false,
            PROFILE,
            geometry,
            mapping(&[("value.input", target)]).unwrap(),
            Some("video"),
        )
    };
    let mut missing = geometry();
    missing.remove(&id("value.output"));
    assert!(call(&missing, "value.joined").is_err());
    let mut changed = geometry();
    changed.get_mut(&id("value.input")).unwrap().dimensions = vec![1, 3];
    assert!(call(&changed, "value.joined").is_err());
    let mut extra = geometry();
    extra.insert(id("value.undeclared"), tensor());
    assert!(call(&extra, "value.joined").is_err());
    for target in ["value.output.video", "value.weight.video"] {
        assert!(call(&geometry(), target).is_err());
    }
    assert!(Fragment::family(&family, &false, PROFILE, &geometry(), BTreeMap::new()).is_err());
    assert!(mapping(&[("value.input", "value.a"), ("value.input", "value.b")]).is_err());
}
