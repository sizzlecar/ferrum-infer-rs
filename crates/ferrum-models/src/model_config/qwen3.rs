//! Qwen3 config: GQA + QK-norm + fused projections.

use ferrum_kernels::backend::{AttnType, MlpType, RopeConfig, TransformerConfig};

use crate::definition::ModelDefinition;

pub fn qwen3_config(def: &ModelDefinition) -> TransformerConfig {
    let num_kv_heads = def.num_key_value_heads.unwrap_or(def.num_attention_heads);
    // Qwen3 has explicit head_dim in config (may differ from hidden_size/num_heads)
    let head_dim = def
        .extra_params
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(def.hidden_size / def.num_attention_heads);

    TransformerConfig {
        num_layers: def.num_hidden_layers,
        hidden_size: def.hidden_size,
        intermediate_size: def.intermediate_size,
        num_heads: def.num_attention_heads,
        num_kv_heads,
        head_dim,
        vocab_size: def.vocab_size,
        max_seq_len: def.max_position_embeddings,
        rms_norm_eps: def.norm_eps as f32,
        rope: RopeConfig {
            theta: def.rope_theta.unwrap_or(1_000_000.0),
            head_dim,
            max_seq_len: def.max_position_embeddings,
        },
        has_qk_norm: true, // Qwen3 always has QK-norm
        attn_type: AttnType::Gqa,
        mlp_type: MlpType::SwiGlu,
    }
}
