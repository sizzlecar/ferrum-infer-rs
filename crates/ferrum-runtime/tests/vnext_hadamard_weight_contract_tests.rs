mod vnext_core_contract;

use std::num::NonZeroU32;

use vnext_core_contract::*;

fn transform() -> HadamardTransformSpec {
    HadamardTransformSpec {
        block_size: NonZeroU32::new(4).unwrap(),
        signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(id(
            "component.signs",
        ))),
        application: HadamardApplication::BeforeMatmul {
            input_permutation: None,
        },
    }
}

fn dense(name: &str) -> PhysicalWeightLayout {
    PhysicalWeightLayout::Dense {
        component_id: id(name),
    }
}

fn rotated(name: &str) -> PhysicalWeightLayout {
    PhysicalWeightLayout::Hadamard {
        values: Box::new(dense(name)),
        transform: transform(),
    }
}

fn schema() -> WeightSchema {
    let mut components = ["qkv", "z", "ba", "embedding"]
        .into_iter()
        .map(|name| WeightComponentSpec {
            id: id(format!("component.{name}")),
            role: WeightComponentRole::Values,
            external_names: vec![format!("{name}.weight")],
            dimensions: vec![2, 8],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        })
        .collect::<Vec<_>>();
    components.push(WeightComponentSpec {
        id: id("component.signs"),
        role: WeightComponentRole::TransformSigns,
        external_names: vec!["metadata.hadamard.signs.8".into()],
        dimensions: vec![8],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    });
    let parts = [
        rotated("component.qkv"),
        rotated("component.z"),
        dense("component.ba"),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, layout)| CompositeWeightPart {
        layout: Box::new(layout),
        logical_offsets: vec![index as u64 * 2, 0],
        extents: vec![2, 8],
    })
    .collect();
    let mut inverse = transform();
    inverse.application = HadamardApplication::AfterEmbeddingLookup;
    WeightSchema {
        format_id: id("weight-format.dense"),
        layout_id: id("weight-layout.hadamard"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![
            WeightTensorSpec {
                id: id("weight.matrix"),
                dimensions: vec![6, 8],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Composite { parts },
                required: true,
            },
            WeightTensorSpec {
                id: id("weight.embedding"),
                dimensions: vec![2, 8],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Hadamard {
                    values: Box::new(dense("component.embedding")),
                    transform: inverse,
                },
                required: true,
            },
        ],
    }
}

fn first_transform(schema: &mut WeightSchema) -> &mut HadamardTransformSpec {
    let PhysicalWeightLayout::Composite { parts } = &mut schema.tensors[0].physical_layout else {
        unreachable!()
    };
    let PhysicalWeightLayout::Hadamard { transform, .. } = parts[0].layout.as_mut() else {
        unreachable!()
    };
    transform
}

#[test]
fn hadamard_preserves_per_projection_basis_and_shares_only_one_sign_component() {
    let source = schema();
    source.validate(&id("family.hadamard")).unwrap();
    let weight = ResolvedWeightBinding::from_schema(&source, &id("weight.matrix")).unwrap();
    assert_eq!(weight.components().len(), 4);
    assert_eq!(
        source.physical_bytes(&id("weight.matrix")).unwrap(),
        3 * 2 * 8 * 4 + 8 * 4
    );
    let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
        unreachable!()
    };
    assert!(matches!(
        parts[0].layout.as_ref(),
        PhysicalWeightLayout::Hadamard { .. }
    ));
    assert!(matches!(
        parts[1].layout.as_ref(),
        PhysicalWeightLayout::Hadamard { .. }
    ));
    assert!(matches!(
        parts[2].layout.as_ref(),
        PhysicalWeightLayout::Dense { .. }
    ));
    let embedding = ResolvedWeightBinding::from_schema(&source, &id("weight.embedding")).unwrap();
    assert_eq!(embedding.components().len(), 2);
    let decoded: ResolvedWeightBinding =
        serde_json::from_slice(&serde_json::to_vec(&weight).unwrap()).unwrap();
    assert_eq!(decoded, weight);
    decoded.validate_logical(&[6, 8], ElementType::F32).unwrap();
}

#[test]
fn hadamard_sign_sharing_does_not_relax_other_component_or_storage_contracts() {
    for mutate in [
        |component: &mut WeightComponentSpec| component.role = WeightComponentRole::Values,
        |component: &mut WeightComponentSpec| component.role = WeightComponentRole::Scales,
        |component: &mut WeightComponentSpec| component.dimensions = vec![2, 4],
        |component: &mut WeightComponentSpec| component.dimensions = vec![4],
        |component: &mut WeightComponentSpec| {
            component.encoding = WeightEncoding::Dense {
                element_type: ElementType::F16,
            }
        },
    ] {
        let mut source = schema();
        mutate(source.components.last_mut().unwrap());
        assert!(source.validate(&id("family.hadamard")).is_err());
    }
    let mut source = schema();
    let HadamardSigns::Explicit(signs) = &mut first_transform(&mut source).signs else {
        unreachable!()
    };
    signs.storage = PhysicalStorageLayout::Strided {
        strides_in_elements: vec![1],
        padding: PhysicalWeightPadding::Exact,
    };
    assert!(source.validate(&id("family.hadamard")).is_err());
    let mut source = schema();
    let HadamardSigns::Explicit(signs) = &mut first_transform(&mut source).signs else {
        unreachable!()
    };
    signs.component_id = id("component.absent");
    assert!(source.validate(&id("family.hadamard")).is_err());

    let mut source = schema();
    let PhysicalWeightLayout::Composite { parts } = &mut source.tensors[0].physical_layout else {
        unreachable!()
    };
    parts[1].layout = parts[0].layout.clone();
    let error = source.validate(&id("family.hadamard")).unwrap_err();
    assert!(error.to_string().contains("referenced more than once"));
}

#[test]
fn hadamard_rejects_invalid_blocks_permutations_rank_and_nested_transforms() {
    for block in [3, 16] {
        let mut source = schema();
        first_transform(&mut source).block_size = NonZeroU32::new(block).unwrap();
        assert!(source.validate(&id("family.hadamard")).is_err());
    }
    for dimensions in [(0, 2, 4), (1, 2, 3), (u64::MAX, 2, 2)] {
        let mut source = schema();
        first_transform(&mut source).application = HadamardApplication::BeforeMatmul {
            input_permutation: Some(GroupedFeatureTranspose {
                inner_extent: dimensions.0,
                first_outer_extent: dimensions.1,
                second_outer_extent: dimensions.2,
            }),
        };
        assert!(source.validate(&id("family.hadamard")).is_err());
    }
    let mut source = schema();
    source.tensors[1].dimensions = vec![16];
    assert!(source.validate(&id("family.hadamard")).is_err());
    let mut source = schema();
    source.tensors[1].dimensions = vec![1, 2, 8];
    source.components[3].dimensions = vec![1, 2, 8];
    assert!(source.validate(&id("family.hadamard")).is_err());
    let mut source = schema();
    let child = source.tensors[0].physical_layout.clone();
    source.tensors[0].physical_layout = PhysicalWeightLayout::Hadamard {
        values: Box::new(child),
        transform: transform(),
    };
    let error = source.validate(&id("family.hadamard")).unwrap_err();
    assert!(error.to_string().contains("nested Hadamard"));
}

#[test]
fn hadamard_fingerprint_and_wire_preserve_every_transform_choice() {
    let source = schema();
    let original = source.fingerprint().unwrap();
    for change in 0..4 {
        let mut candidate = source.clone();
        let transform = first_transform(&mut candidate);
        match change {
            0 => transform.block_size = NonZeroU32::new(8).unwrap(),
            1 => transform.signs = HadamardSigns::Identity,
            2 => transform.application = HadamardApplication::AfterEmbeddingLookup,
            _ => {
                transform.application = HadamardApplication::BeforeMatmul {
                    input_permutation: Some(GroupedFeatureTranspose {
                        inner_extent: 2,
                        first_outer_extent: 2,
                        second_outer_extent: 2,
                    }),
                }
            }
        }
        candidate.validate(&id("family.hadamard")).unwrap();
        assert_ne!(candidate.fingerprint().unwrap(), original);
    }
    let wire = serde_json::to_value(transform()).unwrap();
    let mut invalid = wire.clone();
    invalid["block_size"] = json!(0);
    assert!(serde_json::from_value::<HadamardTransformSpec>(invalid).is_err());
    let mut invalid = wire;
    invalid["hidden_rounding"] = json!("f16");
    assert!(serde_json::from_value::<HadamardTransformSpec>(invalid).is_err());
    assert!(serde_json::from_value::<HadamardApplication>(json!({
        "after_embedding_lookup": { "input_permutation": null }
    }))
    .is_err());
}

#[test]
fn hadamard_identity_omits_sign_storage_and_wrappers_obey_layout_depth_budget() {
    let mut source = schema();
    source.tensors.truncate(1);
    source.components.retain(|component| {
        !matches!(
            component.id.as_str(),
            "component.signs" | "component.embedding"
        )
    });
    let PhysicalWeightLayout::Composite { parts } = &mut source.tensors[0].physical_layout else {
        unreachable!()
    };
    for part in &mut parts[..2] {
        let PhysicalWeightLayout::Hadamard { transform, .. } = part.layout.as_mut() else {
            unreachable!()
        };
        transform.signs = HadamardSigns::Identity;
    }
    source.validate(&id("family.hadamard")).unwrap();
    assert_eq!(
        source.physical_bytes(&id("weight.matrix")).unwrap(),
        3 * 2 * 8 * 4
    );
    for _ in 0..MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH {
        source.tensors[0].physical_layout = PhysicalWeightLayout::Hadamard {
            values: Box::new(source.tensors[0].physical_layout.clone()),
            transform: transform(),
        };
    }
    let error = source.validate(&id("family.hadamard")).unwrap_err();
    assert!(error.to_string().contains("layout depth exceeds"));
}

struct SharedSignsFamily;

impl ModelFamilyProvider for SharedSignsFamily {
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
        Ok(schema())
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        TestFamily.numerical_profiles(config)
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = TestFamily.semantic_program(config, profile)?;
        let mut blocks = base.blocks().to_vec();
        blocks[0].nodes[0].inputs.push(id("value.embedding"));
        let mut weights = base.weights().to_vec();
        weights[0].tensor.dimensions = vec![6, 8];
        weights.push(WeightReference {
            weight_id: id("weight.embedding"),
            value_id: id("value.embedding"),
            tensor: ProgramTensorSpec {
                dimensions: vec![2, 8],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
        ModelProgram::new(
            self.family_id().clone(),
            base.inputs().to_vec(),
            blocks,
            base.states().to_vec(),
            weights,
            base.outputs().to_vec(),
        )
    }
}

#[test]
fn hadamard_shared_signs_use_one_planned_static_range_across_logical_weights() {
    let family = TypedFamilyRegistration::new(SharedSignsFamily)
        .prepare_fixture(&json!({"width": 4}))
        .unwrap();
    let matrix_contract = |rows| {
        TensorContract::new(
            vec![
                DimensionConstraint::Exact(rows),
                DimensionConstraint::Exact(8),
            ],
            BTreeSet::from([ElementType::F32]),
            vec![LayoutConstraint::Contiguous],
            TensorAccess::Read,
            AliasPolicy::NoAlias,
        )
        .unwrap()
    };
    let mut operation = operation();
    operation.inputs[1] = matrix_contract(6);
    operation.inputs.push(matrix_contract(2));
    let catalog = catalog_from_operations(vec![operation.clone()]).unwrap();
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
    let policy = policy(4096);
    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &registry.planning(), &options)
            .unwrap();
    let plan = compilation.executable().execution_plan();
    let weights = plan.payload().nodes()[0]
        .values()
        .iter()
        .filter(|binding| binding.usage() == BufferUsage::Weights)
        .collect::<Vec<_>>();
    assert_eq!(weights.len(), 2);
    let signs = weights
        .iter()
        .map(|weight| {
            weight
                .storage()
                .components()
                .iter()
                .find(|component| component.component_id() == Some(&id("component.signs")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(signs[0], signs[1]);
    assert_eq!(signs[0].length_bytes(), 8 * 4);
    // Four distinct matrices, one shared full-width sign vector, and the
    // estimator's 32-byte persistent workspace. No per-projection sign copy.
    assert_eq!(
        plan.payload().memory().static_bytes(),
        4 * 2 * 8 * 4 + 8 * 4 + 32
    );
    plan.validate_against(&family, &catalog, &policy, compilation.node_resolutions())
        .unwrap();
    ExecutionPlan::from_json_validated(
        &plan.to_json().unwrap(),
        &family,
        &catalog,
        &policy,
        compilation.node_resolutions().to_vec(),
    )
    .unwrap();

    let bindings = plan.payload().nodes()[0].values().to_vec();
    let embedding_index = bindings
        .iter()
        .position(|binding| binding.value_id() == &id("value.embedding"))
        .unwrap();
    let embedding = &bindings[embedding_index];
    let rebind = |access, weight, storage| {
        ResolvedValueBinding::new(
            embedding.value_id().clone(),
            embedding.role(),
            embedding.ordinal(),
            embedding.tensor().clone(),
            access,
            embedding.alias().clone(),
            embedding.usage(),
            Some(weight),
            storage,
        )
        .unwrap()
    };

    let mut components = embedding.storage().components().to_vec();
    let index = components
        .iter()
        .position(|component| component.component_id() == Some(&id("component.signs")))
        .unwrap();
    components[index] = ResolvedStorageComponent::new(
        components[index].component_id().cloned(),
        components[index].resource_id().clone(),
        components[index].offset_bytes() + 4,
        components[index].length_bytes(),
        ElementType::F32,
    )
    .unwrap();
    let mut partial = bindings.clone();
    partial[embedding_index] = rebind(
        TensorAccess::Read,
        embedding.weight().unwrap().clone(),
        ResolvedValueStorage::composite(components).unwrap(),
    );
    assert!(operation.validate_resolved_bindings(&partial).is_err());

    let mut other_schema = schema();
    other_schema.components.last_mut().unwrap().id = id("component.other-signs");
    let PhysicalWeightLayout::Hadamard { transform, .. } =
        &mut other_schema.tensors[1].physical_layout
    else {
        unreachable!()
    };
    transform.signs = HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
        id("component.other-signs"),
    ));
    let other_weight =
        ResolvedWeightBinding::from_schema(&other_schema, &id("weight.embedding")).unwrap();
    let mut components = embedding.storage().components().to_vec();
    components[index] = ResolvedStorageComponent::new(
        Some(id("component.other-signs")),
        components[index].resource_id().clone(),
        components[index].offset_bytes(),
        components[index].length_bytes(),
        ElementType::F32,
    )
    .unwrap();
    let mut different_id = bindings.clone();
    different_id[embedding_index] = rebind(
        TensorAccess::Read,
        other_weight,
        ResolvedValueStorage::composite(components).unwrap(),
    );
    assert!(operation.validate_resolved_bindings(&different_id).is_err());

    let qkv = weights
        .iter()
        .flat_map(|weight| weight.storage().components())
        .find(|component| component.component_id() == Some(&id("component.qkv")))
        .unwrap();
    // An overlapping payload in one logical value must not borrow another
    // component's valid shared-sign relationship in the global alias check.
    let mut overlapping_components = embedding.storage().components().to_vec();
    let payload_index = overlapping_components
        .iter()
        .position(|component| component.component_id() == Some(&id("component.embedding")))
        .unwrap();
    overlapping_components[payload_index] = ResolvedStorageComponent::new(
        overlapping_components[payload_index]
            .component_id()
            .cloned(),
        signs[0].resource_id().clone(),
        signs[0].offset_bytes(),
        overlapping_components[payload_index].length_bytes(),
        ElementType::F32,
    )
    .unwrap();
    assert!(ResolvedValueStorage::composite(overlapping_components).is_err());
    let mut components = embedding.storage().components().to_vec();
    let values_index = components
        .iter()
        .position(|component| component.component_id() == Some(&id("component.embedding")))
        .unwrap();
    components[values_index] = ResolvedStorageComponent::new(
        components[values_index].component_id().cloned(),
        qkv.resource_id().clone(),
        qkv.offset_bytes(),
        qkv.length_bytes(),
        ElementType::F32,
    )
    .unwrap();
    let mut shared_values = bindings.clone();
    shared_values[embedding_index] = rebind(
        TensorAccess::Read,
        embedding.weight().unwrap().clone(),
        ResolvedValueStorage::composite(components).unwrap(),
    );
    assert!(operation
        .validate_resolved_bindings(&shared_values)
        .is_err());

    let mut writable = bindings.clone();
    writable[embedding_index] = rebind(
        TensorAccess::ReadWrite,
        embedding.weight().unwrap().clone(),
        embedding.storage().clone(),
    );
    operation.inputs[embedding.ordinal() as usize] = TensorContract::new(
        vec![DimensionConstraint::Exact(2), DimensionConstraint::Exact(8)],
        BTreeSet::from([ElementType::F32]),
        vec![LayoutConstraint::Contiguous],
        TensorAccess::ReadWrite,
        AliasPolicy::NoAlias,
    )
    .unwrap();
    assert!(operation.validate_resolved_bindings(&writable).is_err());
}
