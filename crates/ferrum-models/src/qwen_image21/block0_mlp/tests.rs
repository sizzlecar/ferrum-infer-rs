use super::*;
use serde_json::json;

fn config() -> QwenImage21Config {
    QwenImage21Config::from_hf_json(&serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel", "num_layers":1, "num_attention_heads":1, "attention_head_dim":6, "context_in_dim":8, "in_channels":3, "out_channels":4, "patch_size":1, "mlp_ratio":3, "axes_dims_rope":[2,2,2], "causal_condition":true, "eps":0.000001})).unwrap(), StructureLimits::default()).unwrap()
}

#[test]
fn mlp_has_three_required_real_weights_and_four_fixed_nodes() {
    let registration = TypedFamilyRegistration::new(QwenImage21Block0MlpFamily::new().unwrap());
    let definition = registration
        .define(&serde_json::to_value(config()).unwrap())
        .unwrap();
    let weights = definition.stored_weight_schema().unwrap();
    assert_eq!(weights.components.len(), 3);
    assert_eq!(weights.tensors.len(), 3);
    assert!(weights
        .components
        .iter()
        .all(|component| component.required));
    let names = weights
        .components
        .iter()
        .flat_map(|component| component.external_names.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        QwenImage21Block0MlpFamily::WEIGHT_NAMES
            .into_iter()
            .collect()
    );
    let prepared = registration
        .prepare(
            &definition,
            &NumericalProfileId::new(QwenImage21Block0MlpFamily::CUDA_PROFILE_ID).unwrap(),
        )
        .unwrap();
    assert_eq!(prepared.metadata(), &ModelSemanticMetadata::TensorComponent);
    let program = prepared.program();
    assert_eq!(
        program.inputs()[0].as_str(),
        QwenImage21Block0MlpFamily::INPUT_ID
    );
    assert_eq!(program.outputs().len(), 4);
    assert!(program.states().is_empty());
    let nodes = &program.blocks()[0].nodes;
    assert_eq!(nodes.len(), 4);
    assert!(nodes
        .iter()
        .all(|node| node.work == ProgramNodeWorkSpec::Fixed));
    assert_eq!(nodes[0].inputs[0], nodes[1].inputs[0]);
    assert_eq!(
        nodes[2].inputs,
        [nodes[0].outputs[0].clone(), nodes[1].outputs[0].clone()]
    );
    assert_eq!(nodes[2].operation_id.as_str(), SILU_MUL_BF16_OPERATION_ID);
    assert_eq!(nodes[3].inputs[0], nodes[2].outputs[0]);
    for (name, expected) in [
        ("transformer_blocks.0.img_mlp.gate_layer.weight", [18, 6]),
        ("transformer_blocks.0.img_mlp.proj.weight", [18, 6]),
        ("transformer_blocks.0.img_mlp.out.weight", [6, 18]),
    ] {
        let id = WeightId::new(format!("weight.qwen_image21.{name}")).unwrap();
        let weight = program
            .weights()
            .iter()
            .find(|weight| weight.weight_id == id)
            .unwrap();
        assert_eq!(weight.tensor.dimensions, expected, "{id}");
    }
}

#[test]
fn mlp_profile_selection_is_explicit_and_no_token_metadata_is_added() {
    let config = config();
    let family = QwenImage21Block0MlpFamily::new().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    assert_eq!(
        profiles.auto_preference(),
        &[NumericalProfileId::new(QwenImage21Block0MlpFamily::REFERENCE_PROFILE_ID).unwrap()]
    );
    for profile in profiles.profiles() {
        assert_eq!(profile.boundaries.len(), 4);
        assert!(profile.kv_storage.is_empty() && profile.states.is_empty());
        assert_eq!(
            family
                .semantic_program(&config, profile)
                .unwrap()
                .weights()
                .len(),
            3
        );
    }
    let mut wrong = profiles.profiles()[0].clone();
    wrong.operations[0].accumulation_type = Some(ElementType::Bf16);
    assert!(family.semantic_program(&config, &wrong).is_err());
}
