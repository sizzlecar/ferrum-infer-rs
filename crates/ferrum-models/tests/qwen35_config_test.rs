use ferrum_models::qwen35_config::{
    Qwen35LayerType, Qwen35MlpKind, Qwen35TextConfig, QWEN35_CONV_STATE_NAME,
    QWEN35_DELTA_STATE_NAME,
};
use ferrum_types::{DataType, Device, RequestId};

const ARTIFACT_ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/goals/model-coverage-2026-06-12/artifacts/",
    "w3_hf_config_probe_20260617T131209Z_f97c1d6f"
);

fn read_artifact(name: &str) -> String {
    std::fs::read_to_string(format!("{ARTIFACT_ROOT}/{name}")).unwrap()
}

#[test]
fn parses_official_qwen35_dense_min_config() {
    let raw = read_artifact("dense_min_reference.config.json");
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
    let specs = cfg.recurrent_state_tensor_specs().unwrap();
    assert_eq!(specs.len(), 36);
    assert_eq!(specs[0].layer_index, 0);
    assert_eq!(specs[1].layer_index, 0);
    assert_eq!(specs[2].layer_index, 1);
    assert_eq!(specs[4].layer_index, 2);
    assert_eq!(specs[6].layer_index, 4);
    assert_eq!(specs[0].name, QWEN35_CONV_STATE_NAME);
    assert_eq!(specs[0].shape, vec![6144, 3]);
    assert_eq!(specs[1].name, QWEN35_DELTA_STATE_NAME);
    assert_eq!(specs[1].shape, vec![16, 128, 128]);
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
    assert_eq!(spec.dtype, DataType::BF16);
    assert_eq!(spec.device, Device::CPU);
    assert_eq!(spec.max_batch_slots, 1);
    assert_eq!(
        spec.estimated_memory_bytes(),
        18 * (6144 * 3 + 16 * 128 * 128) * 2
    );
}

#[test]
fn parses_official_qwen36_shared_expert_moe_config() {
    let raw = read_artifact("moe_shared_expert_reference.config.json");
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
    let specs = cfg.recurrent_state_tensor_specs().unwrap();
    assert_eq!(specs.len(), 60);
    assert_eq!(specs[0].layer_index, 0);
    assert_eq!(specs[1].layer_index, 0);
    assert_eq!(specs[2].layer_index, 1);
    assert_eq!(specs[4].layer_index, 2);
    assert_eq!(specs[6].layer_index, 4);
    assert_eq!(specs[0].name, QWEN35_CONV_STATE_NAME);
    assert_eq!(specs[0].shape, vec![8192, 3]);
    assert_eq!(specs[1].name, QWEN35_DELTA_STATE_NAME);
    assert_eq!(specs[1].shape, vec![32, 128, 128]);
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
        30 * (8192 * 3 + 32 * 128 * 128) * 2
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

    assert_eq!(quant.quant_method, "gptq");
    assert_eq!(quant.bits, 4);
    assert_eq!(quant.group_size, 128);
    assert!(!quant.desc_act);
    assert!(quant.sym);
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
    let raw = read_artifact("dense_min_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    let err = cfg
        .to_recurrent_state_spec(RequestId::new(), DataType::FP16, Device::CPU, 0)
        .expect_err("zero batch slots should fail");
    assert!(err.contains("max_batch_slots"), "{err}");
}

#[test]
fn rejects_out_of_range_layer_plan_lookup() {
    let raw = read_artifact("dense_min_reference.config.json");
    let cfg = Qwen35TextConfig::from_hf_config_str(&raw).unwrap();
    let err = cfg
        .mlp_kind_for_layer(cfg.num_hidden_layers)
        .expect_err("out-of-range layer lookup should fail");
    assert!(err.contains("layer_index"), "{err}");
}
