use super::PH3ImageTransformerConfig;
use serde_json::json;

pub(super) fn tiny_transformer(head_dim: u64, rope_frequencies: u64) -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"MiniMaxH3Transformer3DModel",
            "attention_head_dim":head_dim,"audio_in_channels":2,"ffn_dim":16,
            "final_norm_eps":1e-5,"freq_dim":8,"hidden_size":8,
            "in_channels":2,"norm_eps":1e-5,"num_attention_heads":2,
            "num_layers":1,"num_refiner_layers":1,"patch_size":[1,2,2],
            "qk_norm_eps":1e-5,"rope_freq_dim":rope_frequencies,"rope_theta":10000.0,
            "text_dim":4,"time_embed_dim":4,"time_embed_hidden_dim":8
        }))
        .unwrap(),
    )
    .unwrap()
}
