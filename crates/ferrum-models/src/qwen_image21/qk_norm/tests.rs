use super::*;
use serde_json::json;

fn config() -> QwenImage21Config {
    QwenImage21Config::from_hf_json(&serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel", "num_layers":1, "num_attention_heads":2, "attention_head_dim":6, "context_in_dim":8, "in_channels":3, "out_channels":4, "patch_size":1, "mlp_ratio":3, "axes_dims_rope":[2,2,2], "causal_condition":true, "eps":0.000001})).unwrap(), StructureLimits::default()).unwrap()
}

#[test]
fn partial_family_keeps_real_weights_and_independent_qk_head_norms() {
    let family = QwenImage21Block0QkNormFamily::new().unwrap();
    let config = config();
    let profiles = family.numerical_profiles(&config).unwrap();
    assert_eq!(
        profiles.auto_preference(),
        &[NumericalProfileId::new(QwenImage21Block0QkNormFamily::REFERENCE_PROFILE_ID).unwrap()]
    );
    let registration = TypedFamilyRegistration::new(family);
    let prepared = registration
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(QwenImage21Block0QkNormFamily::CUDA_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let schema = prepared.stored_weight_schema().unwrap();
    assert_eq!(schema.components.len(), 4);
    assert!(schema.components.iter().all(|component| component.required));
    let names = schema
        .components
        .iter()
        .flat_map(|component| component.external_names.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        QwenImage21Block0QkNormFamily::WEIGHT_NAMES
            .into_iter()
            .collect()
    );
    let program = prepared.program();
    assert_eq!(prepared.metadata(), &ModelSemanticMetadata::TensorComponent);
    assert!(program.states().is_empty());
    assert_eq!(program.inputs().len(), 1);
    assert_eq!(program.outputs().len(), 4);
    for (index, name) in QwenImage21Block0QkNormFamily::WEIGHT_NAMES
        .into_iter()
        .enumerate()
    {
        let id = WeightId::new(format!("weight.qwen_image21.{name}")).unwrap();
        let weight = program
            .weights()
            .iter()
            .find(|weight| weight.weight_id == id)
            .unwrap();
        assert_eq!(
            weight.tensor.dimensions,
            if index < 2 { vec![12, 12] } else { vec![6] }
        );
    }
    let nodes = &program.blocks()[0].nodes;
    assert_eq!(nodes.len(), 4);
    for branch in 0..2 {
        assert_eq!(
            nodes[branch].operation_id.as_str(),
            DENSE_LINEAR_BF16_OPERATION_ID
        );
        assert_eq!(
            nodes[branch + 2].operation_id.as_str(),
            HEAD_RMS_NORM_BF16_OPERATION_ID
        );
        assert_eq!(nodes[branch + 2].inputs[0], nodes[branch].outputs[0]);
        assert_eq!(
            nodes[branch + 2].attributes[&AttributeId::new("head_dim").unwrap()],
            SemanticValue::Unsigned(6)
        );
        assert_eq!(nodes[branch + 2].outputs[0], program.outputs()[branch + 2]);
    }
}

#[test]
fn changed_arithmetic_cannot_reuse_the_declared_family_profile() {
    let family = QwenImage21Block0QkNormFamily::new().unwrap();
    let config = config();
    let profiles = family.numerical_profiles(&config).unwrap();
    let mut profile = profiles.profiles()[0].clone();
    profile.operations[1].accumulation_type = Some(ElementType::Bf16);
    assert!(family.semantic_program(&config, &profile).is_err());
}
