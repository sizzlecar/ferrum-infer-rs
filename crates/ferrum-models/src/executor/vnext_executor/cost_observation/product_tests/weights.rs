//! Small deterministic Qwen35 recurrent+causal safetensors; no downloaded assets.
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::path::Path;

// The Metal GDN scratch ABI packs each row to 16 bytes: half activation rows
// require multiples of 8 elements and the per-value-head float rows require
// multiples of 4. Keep this fixture tiny without violating that real ABI.
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 8;
const KEY_HEADS: usize = 2;
const VALUE_HEADS: usize = 4;
const HEAD_DIM: usize = 4;
const VALUE_FEATURES: usize = VALUE_HEADS * HEAD_DIM;
const QKV_FEATURES: usize = 2 * KEY_HEADS * HEAD_DIM + VALUE_FEATURES;

#[derive(Clone, Copy)]
pub(super) struct CausalGeometry {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub gate: bool,
}
impl CausalGeometry {
    pub const TINY: Self = Self {
        heads: 1,
        kv_heads: 1,
        head_dim: HIDDEN,
        context: 16,
        gate: false,
    };
    pub const GROUPED: Self = Self {
        heads: 16,
        kv_heads: 4,
        head_dim: 256,
        context: 512,
        gate: true,
    };
}
pub(super) fn write_config(dir: &Path, causal: CausalGeometry) {
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vocab_size": 3,
        "max_position_embeddings": causal.context,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": KEY_HEADS,
            "linear_num_value_heads": VALUE_HEADS,
            "linear_key_head_dim": HEAD_DIM,
            "linear_value_head_dim": HEAD_DIM,
            "linear_conv_kernel_dim": 2,
            "mamba_ssm_dtype": "float32",
            "head_dim": causal.head_dim,
            "num_attention_heads": causal.heads,
            "num_key_value_heads": causal.kv_heads,
            "attn_output_gate": causal.gate,
            "vocab_size": 3,
            "max_position_embeddings": causal.context,
            "tie_word_embeddings": false
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

pub(super) fn write_weights(dir: &Path, causal: CausalGeometry) {
    let tensors: Vec<(String, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        ("model.norm.weight".to_string(), vec![0.0, 0.0]),
        (
            "model.lm_head.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            vec![1.0, -1.0],
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            vec![0.5, 0.25],
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            vec![-0.25, 0.75],
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ),
        ("model.layers.0.linear_attn.A_log".to_string(), vec![0.0]),
        ("model.layers.0.linear_attn.dt_bias".to_string(), vec![0.0]),
        (
            "model.layers.0.linear_attn.norm.weight".to_string(),
            vec![1.0],
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight".to_string(),
            vec![1.0, -0.5],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![0.2, 0.1, -0.1, 0.3],
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            vec![0.4, -0.2, 0.3, 0.5],
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.input_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.1.post_attention_layernorm.weight".to_string(),
            vec![0.0, 0.0],
        ),
        (
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.k_proj.weight".to_string(),
            vec![0.5, 0.0, 0.0, 0.5],
        ),
        (
            "model.layers.1.self_attn.v_proj.weight".to_string(),
            vec![1.0, 1.0, -0.5, 0.5],
        ),
        (
            "model.layers.1.self_attn.o_proj.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.q_norm.weight".to_string(),
            vec![1.0, 1.0],
        ),
        (
            "model.layers.1.self_attn.k_norm.weight".to_string(),
            vec![1.0, 1.0],
        ),
        (
            "model.layers.1.mlp.gate_proj.weight".to_string(),
            vec![-0.2, 0.2, 0.1, 0.3],
        ),
        (
            "model.layers.1.mlp.up_proj.weight".to_string(),
            vec![0.25, 0.5, -0.3, 0.4],
        ),
        (
            "model.layers.1.mlp.down_proj.weight".to_string(),
            vec![0.5, 0.25, -0.2, 0.75],
        ),
    ];
    let views = tensors
        .into_iter()
        .map(|(name, values)| {
            let dimensions = if name.ends_with("embed_tokens.weight")
                || name.ends_with("lm_head.weight")
            {
                vec![3, HIDDEN]
            } else if name.ends_with("in_proj_qkv.weight") {
                vec![QKV_FEATURES, HIDDEN]
            } else if name.ends_with("in_proj_z.weight") {
                vec![VALUE_FEATURES, HIDDEN]
            } else if ["in_proj_b.weight", "in_proj_a.weight"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                vec![VALUE_HEADS, HIDDEN]
            } else if name.ends_with("linear_attn.out_proj.weight") {
                vec![HIDDEN, VALUE_FEATURES]
            } else if name.ends_with("conv1d.weight") {
                vec![QKV_FEATURES, 1, 2]
            } else if name.ends_with("linear_attn.A_log") || name.ends_with("linear_attn.dt_bias") {
                vec![VALUE_HEADS]
            } else if name.ends_with("linear_attn.norm.weight") {
                vec![HEAD_DIM]
            } else if name.ends_with("mlp.down_proj.weight") {
                vec![HIDDEN, INTERMEDIATE]
            } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight")
            {
                vec![INTERMEDIATE, HIDDEN]
            } else if name.ends_with("self_attn.q_proj.weight") {
                vec![
                    causal.heads * causal.head_dim * if causal.gate { 2 } else { 1 },
                    HIDDEN,
                ]
            } else if name.ends_with("self_attn.k_proj.weight")
                || name.ends_with("self_attn.v_proj.weight")
            {
                vec![causal.kv_heads * causal.head_dim, HIDDEN]
            } else if name.ends_with("self_attn.o_proj.weight") {
                vec![HIDDEN, causal.heads * causal.head_dim]
            } else if name.ends_with("self_attn.q_norm.weight")
                || name.ends_with("self_attn.k_norm.weight")
            {
                vec![causal.head_dim]
            } else {
                vec![HIDDEN]
            };
            // Repeat the deterministic nonzero seed patterns into the actual
            // declared shape. Both guarded and ordinary baselines load these
            // same dense tensors; no provider or numerical contract is mocked.
            let bytes = values
                .iter()
                .cycle()
                .take(dimensions.iter().product())
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
                .into_boxed_slice();
            (name, dimensions, bytes)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        views.iter().map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        &None::<std::collections::HashMap<String, String>>,
        &dir.join("model.safetensors"),
    )
    .unwrap();
}
