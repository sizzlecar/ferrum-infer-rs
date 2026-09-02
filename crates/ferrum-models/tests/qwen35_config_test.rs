use ferrum_models::qwen35_config::{
    Qwen35Fp8ActivationScheme, Qwen35Fp8Format, Qwen35Fp8WeightBlockSize, Qwen35LayerType,
    Qwen35MlpKind, Qwen35TextConfig, QWEN35_CONV_STATE_NAME, QWEN35_DELTA_STATE_NAME,
};
use ferrum_types::{DataType, Device, RequestId};

const ARTIFACT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

const QWEN38_FP8_CONFIG: &str = include_str!("fixtures/qwen38_fp8_config.contract.json");
const QWEN38_FP8_BAD_RECIPE: &str = include_str!("fixtures/qwen38_fp8_config.bad-recipe.json");

fn read_artifact(name: &str) -> String {
    std::fs::read_to_string(format!("{ARTIFACT_ROOT}/{name}")).unwrap()
}

#[test]
fn parses_official_qwen35_dense_min_config() {
    let raw = read_artifact("qwen35_dense_min_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    assert!(!cfg.is_moe());
    assert_eq!(cfg.top_level_model_type.as_deref(), Some("qwen3_5"));
    assert_eq!(cfg.text_model_type, "qwen3_5_text");
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.linear_attention_layers(), 18);
    assert_eq!(cfg.full_attention_layers(), 6);
    assert_eq!(cfg.first_linear_attention_layer(), Some(0));
    assert_eq!(cfg.first_full_attention_layer(), Some(3));
    assert_eq!(cfg.linear_attention.num_key_heads, 16);
    assert_eq!(cfg.linear_attention.num_value_heads, 16);
    assert_eq!(cfg.linear_attention.key_head_dim, 128);
    assert_eq!(cfg.linear_attention.value_head_dim, 128);
    assert_eq!(cfg.linear_attention.conv_kernel_dim, 4);
    assert_eq!(cfg.mamba_ssm_dtype, DataType::FP32);
    assert!(cfg.attn_output_gate);
    assert_eq!(cfg.rope_parameters.rope_theta, 10_000_000.0);
    assert_eq!(cfg.rope_parameters.partial_rotary_factor, 0.25);
    assert!(cfg.rope_parameters.mrope_interleaved);
    assert_eq!(cfg.full_attention_query_total_dim(), 2048);
    assert_eq!(cfg.full_attention_kv_total_dim(), 512);
    assert_eq!(cfg.full_attention_q_proj_total_dim(), 4096);
    assert_eq!(cfg.full_attention_rope_dim(), 64);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.dense_intermediate_size, Some(3584));
    assert_eq!(cfg.dense_mlp_layers().len(), 24);
    assert!(cfg.sparse_moe_layers().is_empty());
    let plan = cfg.layer_plan().unwrap();
    assert_eq!(plan.len(), 24);
    assert_eq!(plan[0].layer_index, 0);
    assert_eq!(plan[0].attention, Qwen35LayerType::LinearAttention);
    assert_eq!(plan[0].mlp, Qwen35MlpKind::Dense);
    assert!(plan[0].has_recurrent_state);
    assert_eq!(plan[3].attention, Qwen35LayerType::FullAttention);
    assert_eq!(plan[3].mlp, Qwen35MlpKind::Dense);
    assert!(!plan[3].has_recurrent_state);
    assert_eq!(cfg.linear_qk_total_dim(), 2048);
    assert_eq!(cfg.linear_value_total_dim(), 2048);
    let manifest = cfg.weight_manifest("model").unwrap();
    assert_eq!(
        manifest
            .global_tensors
            .iter()
            .find(|tensor| tensor.role == "lm_head")
            .unwrap()
            .required,
        false
    );
    assert!(manifest.layers[0]
        .tensors
        .iter()
        .any(|tensor| tensor.name == "model.layers.0.linear_attn.in_proj_qkv.weight"));
    assert!(manifest.layers[0]
        .tensors
        .iter()
        .any(|tensor| tensor.name == "model.layers.0.mlp.gate_proj.weight"));
    assert!(manifest.layers[3]
        .tensors
        .iter()
        .any(|tensor| tensor.name == "model.layers.3.self_attn.q_proj.weight"));
    assert_eq!(
        cfg.recurrent_delta_state_shape().unwrap(),
        vec![16, 128, 128]
    );
    assert_eq!(cfg.recurrent_conv_state_shape().unwrap(), vec![6144, 3]);
    let specs = cfg.recurrent_state_tensor_specs(DataType::BF16).unwrap();
    assert_eq!(specs.len(), 36);
    assert_eq!(specs[0].layer_index, 0);
    assert_eq!(specs[1].layer_index, 0);
    assert_eq!(specs[2].layer_index, 1);
    assert_eq!(specs[4].layer_index, 2);
    assert_eq!(specs[6].layer_index, 4);
    assert_eq!(specs[0].name, QWEN35_CONV_STATE_NAME);
    assert_eq!(specs[0].shape, vec![6144, 3]);
    assert_eq!(specs[0].dtype, DataType::BF16);
    assert_eq!(specs[1].name, QWEN35_DELTA_STATE_NAME);
    assert_eq!(specs[1].shape, vec![16, 128, 128]);
    assert_eq!(specs[1].dtype, DataType::FP32);
    assert_eq!(
        cfg.recurrent_state_elements_per_slot().unwrap(),
        18 * (6144 * 3 + 16 * 128 * 128)
    );
    let request_id = RequestId::new();
    let spec = cfg
        .to_recurrent_state_spec(request_id.clone(), DataType::BF16, Device::CPU, 1)
        .unwrap();
    assert_eq!(spec.request_id, request_id);
    assert_eq!(spec.num_layers, 24);
    assert_eq!(spec.tensors.len(), 36);
    assert_eq!(spec.tensors[0].shape, vec![6144, 3]);
    assert_eq!(spec.tensors[1].shape, vec![16, 128, 128]);
    assert_eq!(spec.device, Device::CPU);
    assert_eq!(spec.max_batch_slots, 1);
    assert_eq!(
        spec.estimated_memory_bytes(),
        18 * (6144 * 3 * 2 + 16 * 128 * 128 * 4)
    );
}

#[test]
fn rejects_missing_or_unsupported_mamba_ssm_dtype() {
    let raw = read_artifact("qwen35_dense_min_reference.config.json");
    let mut missing: serde_json::Value = serde_json::from_str(&raw).unwrap();
    missing["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("mamba_ssm_dtype");
    let error = Qwen35TextConfig::from_hf_config_value(&missing)
        .expect_err("missing temporal state dtype must fail closed");
    assert!(
        error.contains("mamba_ssm_dtype must be a string"),
        "{error}"
    );

    let mut unsupported: serde_json::Value = serde_json::from_str(&raw).unwrap();
    unsupported["text_config"]["mamba_ssm_dtype"] = serde_json::json!("float8");
    let error = Qwen35TextConfig::from_hf_config_value(&unsupported)
        .expect_err("unsupported temporal state dtype must fail closed");
    assert!(
        error.contains("unsupported Qwen3.5 mamba_ssm_dtype"),
        "{error}"
    );
}

#[test]
fn parses_official_qwen36_shared_expert_moe_config() {
    let raw = read_artifact("qwen35_moe_shared_expert_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    assert!(cfg.is_moe());
    assert!(cfg.quantization.is_none());
    assert_eq!(cfg.top_level_model_type.as_deref(), Some("qwen3_5_moe"));
    assert_eq!(cfg.text_model_type, "qwen3_5_moe_text");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.linear_attention_layers(), 30);
    assert_eq!(cfg.full_attention_layers(), 10);
    assert_eq!(cfg.layer_types[3], Qwen35LayerType::FullAttention);
    assert_eq!(cfg.linear_attention.num_key_heads, 16);
    assert_eq!(cfg.linear_attention.num_value_heads, 32);
    assert_eq!(cfg.mamba_ssm_dtype, DataType::FP32);
    assert!(cfg.attn_output_gate);
    assert_eq!(cfg.rope_parameters.rope_theta, 10_000_000.0);
    assert_eq!(cfg.rope_parameters.partial_rotary_factor, 0.25);
    assert!(cfg.rope_parameters.mrope_interleaved);
    assert_eq!(cfg.full_attention_query_total_dim(), 4096);
    assert_eq!(cfg.full_attention_kv_total_dim(), 512);
    assert_eq!(cfg.full_attention_q_proj_total_dim(), 8192);
    assert_eq!(cfg.full_attention_rope_dim(), 64);
    assert!(!cfg.tie_word_embeddings);
    let moe = cfg.moe.as_ref().unwrap();
    assert_eq!(moe.num_experts, 256);
    assert_eq!(moe.num_experts_per_tok, 8);
    assert_eq!(moe.moe_intermediate_size, 512);
    assert_eq!(moe.shared_expert_intermediate_size, 512);
    assert!(moe.norm_topk_prob);
    assert!(cfg.dense_mlp_layers().is_empty());
    assert_eq!(cfg.sparse_moe_layers().len(), 40);
    assert_eq!(cfg.sparse_moe_layers()[0], 0);
    assert_eq!(cfg.sparse_moe_layers()[39], 39);
    let plan = cfg.layer_plan().unwrap();
    assert_eq!(plan.len(), 40);
    assert_eq!(plan[0].attention, Qwen35LayerType::LinearAttention);
    assert_eq!(plan[0].mlp, Qwen35MlpKind::SparseMoeSharedExpert);
    assert!(plan[0].has_recurrent_state);
    assert_eq!(plan[3].attention, Qwen35LayerType::FullAttention);
    assert_eq!(plan[3].mlp, Qwen35MlpKind::SparseMoeSharedExpert);
    assert!(!plan[3].has_recurrent_state);
    assert_eq!(cfg.linear_qk_total_dim(), 2048);
    assert_eq!(cfg.linear_value_total_dim(), 4096);
    let manifest = cfg.weight_manifest("model.language_model").unwrap();
    assert_eq!(
        manifest
            .global_tensors
            .iter()
            .find(|tensor| tensor.role == "lm_head")
            .unwrap()
            .required,
        true
    );
    assert!(manifest.layers[0].tensors.iter().any(
        |tensor| tensor.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    ));
    assert!(manifest.layers[0]
        .tensors
        .iter()
        .any(|tensor| tensor.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"));
    assert!(manifest.layers[3]
        .tensors
        .iter()
        .any(|tensor| tensor.name == "model.language_model.layers.3.self_attn.q_proj.weight"));
    assert!(manifest.layers[3]
        .tensors
        .iter()
        .any(|tensor| tensor.name
            == "model.language_model.layers.3.mlp.shared_expert.down_proj.weight"));
    assert_eq!(
        cfg.recurrent_delta_state_shape().unwrap(),
        vec![32, 128, 128]
    );
    assert_eq!(cfg.recurrent_conv_state_shape().unwrap(), vec![8192, 3]);
    let specs = cfg.recurrent_state_tensor_specs(DataType::FP16).unwrap();
    assert_eq!(specs.len(), 60);
    assert_eq!(specs[0].layer_index, 0);
    assert_eq!(specs[1].layer_index, 0);
    assert_eq!(specs[2].layer_index, 1);
    assert_eq!(specs[4].layer_index, 2);
    assert_eq!(specs[6].layer_index, 4);
    assert_eq!(specs[0].name, QWEN35_CONV_STATE_NAME);
    assert_eq!(specs[0].shape, vec![8192, 3]);
    assert_eq!(specs[0].dtype, DataType::FP16);
    assert_eq!(specs[1].name, QWEN35_DELTA_STATE_NAME);
    assert_eq!(specs[1].shape, vec![32, 128, 128]);
    assert_eq!(specs[1].dtype, DataType::FP32);
    assert_eq!(
        cfg.recurrent_state_elements_per_slot().unwrap(),
        30 * (8192 * 3 + 32 * 128 * 128)
    );
    let spec = cfg
        .to_recurrent_state_spec(RequestId::new(), DataType::FP16, Device::CPU, 1)
        .unwrap();
    assert_eq!(spec.num_layers, 40);
    assert_eq!(spec.tensors.len(), 60);
    assert_eq!(spec.tensors[0].shape, vec![8192, 3]);
    assert_eq!(spec.tensors[1].shape, vec![32, 128, 128]);
    assert_eq!(
        spec.estimated_memory_bytes(),
        30 * (8192 * 3 * 2 + 32 * 128 * 128 * 4)
    );
}

#[test]
fn parses_qwen35_moe_gptq_quantization_config() {
    let raw = r#"{
      "model_type": "qwen3_5_moe",
      "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "desc_act": false,
        "sym": true,
        "quant_method": "gptq"
      },
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "head_dim": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "attn_output_gate": true,
        "rope_parameters": {
          "rope_theta": 10000000,
          "partial_rotary_factor": 0.25,
          "mrope_interleaved": true
        },
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": false
      }
    }"#;
    let cfg = Qwen35TextConfig::from_hf_config_str(raw).unwrap();
    let quant = cfg
        .quantization
        .as_ref()
        .expect("GPTQ quantization_config should be preserved");
    let recipe = quant.as_gptq().expect("typed GPTQ recipe");

    assert_eq!(quant.quant_method(), "gptq");
    assert_eq!(recipe.bits, 4);
    assert_eq!(recipe.group_size, 128);
    assert!(!recipe.desc_act);
    assert!(recipe.sym);
}

#[test]
fn parses_fixed_qwen38_official_block_fp8_recipe_without_legacy_placeholders() {
    let cfg = Qwen35TextConfig::from_hf_config_str(QWEN38_FP8_CONFIG).unwrap();
    let quantization = cfg.quantization.as_ref().expect("typed FP8 metadata");
    let recipe = quantization.as_fp8().expect("official block-FP8 recipe");

    assert_eq!(quantization.quant_method(), "fp8");
    assert_eq!(recipe.format, Qwen35Fp8Format::E4m3);
    assert_eq!(recipe.activation_scheme, Qwen35Fp8ActivationScheme::Dynamic);
    assert_eq!(
        recipe.weight_block_size,
        Qwen35Fp8WeightBlockSize::OFFICIAL_128X128
    );
    assert_eq!(recipe.weight_block_size.as_array(), [128, 128]);
    assert_eq!(recipe.modules_to_not_convert.len(), 10);
    assert!(recipe
        .modules_to_not_convert
        .contains(&"model.visual.blocks.0.attn.proj".to_string()));
    assert!(recipe
        .modules_to_not_convert
        .contains(&"model.language_model.layers.0.linear_attn.conv1d".to_string()));
    assert!(recipe
        .modules_to_not_convert
        .contains(&"mtp.pre_fc_norm_hidden".to_string()));
}

#[test]
fn rejects_qwen38_fp8_metadata_mismatch_fixture_before_source_loading() {
    let mutation: serde_json::Value = serde_json::from_str(QWEN38_FP8_BAD_RECIPE).unwrap();
    let mut candidate: serde_json::Value = serde_json::from_str(QWEN38_FP8_CONFIG).unwrap();
    let pointer = mutation["pointer"].as_str().unwrap();
    *candidate.pointer_mut(pointer).unwrap() = mutation["replacement"].clone();

    let error = Qwen35TextConfig::from_hf_config_value(&candidate)
        .expect_err("mismatched FP8 format must fail before source loading");
    assert!(
        error.contains(mutation["expected_error"].as_str().unwrap()),
        "{error}"
    );
}

#[test]
fn rejects_missing_unknown_and_wrong_block_fp8_recipe_fields() {
    let fixture: serde_json::Value = serde_json::from_str(QWEN38_FP8_CONFIG).unwrap();

    let mut missing = fixture.clone();
    missing["quantization_config"]
        .as_object_mut()
        .unwrap()
        .remove("activation_scheme");
    let error = Qwen35TextConfig::from_hf_config_value(&missing)
        .expect_err("missing FP8 activation scheme must fail closed");
    assert!(
        error.contains("activation_scheme must be a string"),
        "{error}"
    );

    let mut unknown = fixture.clone();
    unknown["quantization_config"]["checkpoint_format"] = serde_json::json!("fp8");
    let error = Qwen35TextConfig::from_hf_config_value(&unknown)
        .expect_err("unknown FP8 metadata must fail closed");
    assert!(
        error.contains("unsupported Qwen3.5 FP8 quantization_config field"),
        "{error}"
    );

    let mut wrong_block = fixture;
    wrong_block["quantization_config"]["weight_block_size"] = serde_json::json!([64, 128]);
    let error = Qwen35TextConfig::from_hf_config_value(&wrong_block)
        .expect_err("unsupported FP8 block shape must fail closed");
    assert!(
        error.contains("unsupported Qwen3.5 FP8 weight_block_size"),
        "{error}"
    );
}

#[test]
fn rejects_unknown_qwen35_quantization_method() {
    let raw = r#"{
      "model_type": "qwen3_5_moe",
      "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq"
      },
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4
      }
    }"#;
    let err = Qwen35TextConfig::from_hf_config_str(raw)
        .expect_err("unsupported quantization method should fail");
    assert!(err.contains("quant_method"), "{err}");
    assert!(err.contains("awq"), "{err}");
}

#[test]
fn rejects_dense_config_with_moe_fields() {
    let raw = r#"{
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 32,
        "num_experts": 8
      }
    }"#;
    let err = Qwen35TextConfig::from_hf_config_str(raw)
        .expect_err("dense config with MoE fields should fail");
    assert!(err.contains("num_experts"), "{err}");
}

#[test]
fn rejects_moe_config_without_shared_expert() {
    let raw = r#"{
      "model_type": "qwen3_5_moe",
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4
      }
    }"#;
    let err = Qwen35TextConfig::from_hf_config_str(raw)
        .expect_err("MoE config without shared expert should fail");
    assert!(err.contains("shared_expert_intermediate_size"), "{err}");
}

#[test]
fn rejects_zero_recurrent_state_batch_slots() {
    let raw = read_artifact("qwen35_dense_min_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    let err = cfg
        .to_recurrent_state_spec(RequestId::new(), DataType::FP16, Device::CPU, 0)
        .expect_err("zero batch slots should fail");
    assert!(err.contains("max_batch_slots"), "{err}");
}

#[test]
fn rejects_recurrent_state_dimension_overflow_before_spec_creation() {
    let raw = read_artifact("qwen35_dense_min_reference.config.json");
    let mut cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    cfg.linear_attention.num_key_heads = usize::MAX;

    let err = cfg
        .recurrent_state_tensor_specs(DataType::FP16)
        .expect_err("overflowing state dimensions must fail before allocation");

    assert!(err.contains("QK width overflows"), "{err}");
}

#[test]
fn rejects_out_of_range_layer_plan_lookup() {
    let raw = read_artifact("qwen35_dense_min_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    let err = cfg
        .mlp_kind_for_layer(cfg.num_hidden_layers)
        .expect_err("out-of-range layer lookup should fail");
    assert!(err.contains("layer_index"), "{err}");
}
