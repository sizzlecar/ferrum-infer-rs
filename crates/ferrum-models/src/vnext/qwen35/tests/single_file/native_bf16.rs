use super::*;

#[test]
fn explicit_default_precision_preserves_existing_definition_identity() {
    let fixture = Fixture::new();
    let sources = Arc::new(fixture.open().unwrap());
    let existing = define_from_sources(sources.clone()).unwrap();
    let explicit =
        define_from_sources_with_dense_precision(sources, Qwen35DensePrecision::F16).unwrap();
    assert_eq!(
        existing.definition().fingerprint().unwrap(),
        explicit.definition().fingerprint().unwrap()
    );
    assert_eq!(existing.descriptor(), explicit.descriptor());
    assert!(existing
        .definition()
        .canonical_config()
        .get("dense_precision")
        .is_none());
    let parsed: Qwen35FamilyConfig =
        serde_json::from_value(existing.definition().canonical_config().clone()).unwrap();
    assert_eq!(parsed.dense_precision, Qwen35DensePrecision::F16);
    assert_eq!(
        serde_json::to_value(parsed).unwrap(),
        *existing.definition().canonical_config()
    );
}

#[test]
fn native_bf16_definition_materializes_original_weights_and_distinct_profiles() {
    let fixture = Fixture::new();
    let sources = Arc::new(fixture.open().unwrap());
    let old = define_from_sources(sources.clone()).unwrap();
    let native =
        define_from_sources_with_dense_precision(sources, Qwen35DensePrecision::Bf16).unwrap();
    assert_ne!(
        old.definition().fingerprint().unwrap(),
        native.definition().fingerprint().unwrap()
    );
    assert_eq!(
        native.definition().canonical_config()["dense_precision"],
        "bf16"
    );
    assert!(old
        .prepare(&BF16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .is_err());
    assert!(native
        .prepare(&F16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .is_err());
    assert!(native
        .prepare(&F32_MASTER_NUMERICAL_PROFILE_ID.parse().unwrap())
        .is_err());
    let prepared = native
        .prepare(&BF16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let schema = prepared.family().stored_weight_schema().unwrap();
    for component in &schema.components {
        let f32 = component.external_names.iter().any(|name| {
            name.ends_with("linear_attn.A_log")
                || name.ends_with("linear_attn.dt_bias")
                || name.ends_with("linear_attn.norm.weight")
        });
        assert_eq!(
            component.dense_element_type(),
            Some(if f32 {
                ElementType::F32
            } else {
                ElementType::Bf16
            }),
            "{}",
            component.id
        );
    }
    let embedding = schema
        .components
        .iter()
        .find(|c| c.external_names == ["model.language_model.embed_tokens.weight"])
        .unwrap();
    assert_eq!(
        prepared.weights().component(embedding).unwrap().bytes(),
        bf16::from_f32(0.375).to_le_bytes().repeat(32 * 16)
    );
    let norm = schema
        .components
        .iter()
        .find(|c| c.external_names == ["model.language_model.norm.weight"])
        .unwrap();
    assert_eq!(
        prepared.weights().component(norm).unwrap().bytes(),
        bf16::from_f32(1.125).to_le_bytes().repeat(16)
    );
    assert_eq!(native.descriptor().execution_dtype(), DataType::BF16);
    let program = prepared.family().program();
    for state in program.states() {
        assert_eq!(
            state.tensor.element_type,
            if state.id.as_str().ends_with(".delta") {
                ElementType::F32
            } else {
                ElementType::Bf16
            }
        );
    }
    for weight in program.weights() {
        let role = weight.weight_id.as_str();
        let f32 = [
            "linear_attn_a_log",
            "linear_attn_dt_bias",
            "linear_attn_norm",
        ]
        .iter()
        .any(|suffix| role.ends_with(suffix));
        assert_eq!(
            weight.tensor.element_type,
            if f32 {
                ElementType::F32
            } else {
                ElementType::Bf16
            },
            "{role}"
        );
    }
    let profile = prepared.family().numerical_profile();
    assert!(profile
        .kv_storage
        .iter()
        .all(|state| state.format() == KvStorageFormat::Bf16));
    assert_eq!(
        profile.boundaries[&value_id("value.output.logits").unwrap()],
        ElementType::Bf16
    );
    assert_eq!(
        profile.boundaries[&value_id("value.output.greedy_token").unwrap()],
        ElementType::U32
    );
    assert!(program.blocks()[0]
        .nodes
        .iter()
        .all(|n| n.operation_id.as_str().ends_with(".bf16")));
}

#[test]
fn native_bf16_visual_program_preserves_recurrent_state_and_declares_bf16_selection() {
    let mut config = test_config();
    config.dense_precision = Qwen35DensePrecision::Bf16;
    let provider = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = provider
        .define(&serde_json::to_value(config).unwrap())
        .unwrap();
    let visual = provider
        .prepare(
            &definition,
            &BF16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID.parse().unwrap(),
        )
        .unwrap();
    let nodes = &visual.program().blocks()[0].nodes;
    assert_eq!(
        nodes[1].operation_id.as_str(),
        ferrum_runtime::ROW_SELECT_BF16_OPERATION_ID
    );
    let full = nodes
        .iter()
        .find(|n| {
            n.operation_id.as_str()
                == ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_OPERATION_ID
        })
        .unwrap();
    assert_eq!(full.inputs[9].as_str(), ROPE_COS_INPUT_ID);
    assert_eq!(full.inputs[10].as_str(), ROPE_SIN_INPUT_ID);
    assert!(visual
        .program()
        .states()
        .iter()
        .all(|s| s.checkpoint == ferrum_runtime::StateCheckpointCapability::Unsupported));
    assert!(visual
        .program()
        .states()
        .iter()
        .filter(|s| s.id.as_str().ends_with(".delta"))
        .all(|s| s.tensor.element_type == ElementType::F32));
}

#[test]
fn native_bf16_rejects_quantized_and_moe_physical_definitions() {
    let provider = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    for mut config in [
        test_dense_gguf_config(),
        test_block_fp8_config(),
        test_moe_gptq_config(),
    ] {
        config.dense_precision = Qwen35DensePrecision::Bf16;
        assert!(provider
            .define(&serde_json::to_value(config).unwrap())
            .is_err());
    }
}
